//! The exchange's rows and calls over a faked NCCL and driver, and its reduce-scatter's failures against the sum's.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const chunk = @import("cuda_chunk.zig");
const mtp = @import("cuda_mtp.zig");
const exchange = @import("cuda_moe_exchange.zig");
const Collective = @import("cuda_collective.zig").Collective;
const nccl = cuda.nccl;
const abi = cuda.abi;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

const dims = wt.dims;
const slots = wt.slots;
const row_y = slots * dims * 4;

// Each NCCL call and launch in order; an asynchronous state the waits read.
const Fake = struct {
    const Call = struct { kind: enum { scatter, gather, reduce, launch }, send: u64 = 0, recv: u64 = 0, count: usize = 0, f32: bool = true, grid: [6]u32 = @splat(0), args: [5]u64 = @splat(0) };
    var calls: [4]Call = undefined;
    var len: usize = 0;
    var state: nccl.Result = 0;
    var aborts: usize = 0;
    var polls: usize = 0;

    fn reset() void {
        len = 0;
        state = 0;
        aborts = 0;
        polls = 0;
    }
    fn record(c: Call) void {
        calls[len] = c;
        len += 1;
    }
    fn scatter(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, op: nccl.RedOp, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .scatter, .send = send, .recv = recv, .count = count, .f32 = t == .f32 and op == .sum });
        return 0;
    }
    fn reduce(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, op: nccl.RedOp, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .reduce, .send = send, .recv = recv, .count = count, .f32 = t == .f32 and op == .sum });
        return 0;
    }
    fn gather(send: abi.DevicePtr, recv: abi.DevicePtr, count: usize, t: nccl.DataType, _: nccl.Comm, _: abi.Stream) callconv(.c) nccl.Result {
        record(.{ .kind = .gather, .send = send, .recv = recv, .count = count, .f32 = t == .f32 });
        return 0;
    }
    fn read(comptime T: type, args: ?[*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[i].?))).*;
    }
    fn launch(_: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, _: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        var c: Call = .{ .kind = .launch, .grid = .{ gx, gy, gz, bx, by, bz } };
        for (0..3) |i| c.args[i] = read(u64, args, i);
        for (3..5) |i| c.args[i] = @intCast(read(c_int, args, i));
        record(c);
        return 0;
    }
    fn asyncError(_: nccl.Comm, out: *nccl.Result) callconv(.c) nccl.Result {
        polls += 1;
        out.* = state;
        return 0;
    }
    fn abort(_: nccl.Comm) callconv(.c) nccl.Result {
        aborts += 1;
        return 0;
    }
};

const Bufs = struct { y: u64 = 0x1000_0000, wts: u64 = 0x2000_0000, moe_f32: u64 = 0x3000_0000 };

const Pass = struct {
    s: *const Bufs,
    k: struct { moe_sum: cuda.Function },
    stream: cuda.Stream,
    pending_f32: bool = false,
};

/// Rank `rank`'s collective over one end of `pair`, its NCCL faked.
fn collective(pair: *[2]c_int, rank: usize, world: usize, timeout_ms: u64) !Collective {
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, pair) != 0) return error.SocketPair;
    var c: Collective = undefined;
    c.failed = false;
    c.comm = @ptrFromInt(0x1000);
    c.link = .{ .fd = pair[0] };
    c.timeout_ms = timeout_ms;
    c.rank = rank;
    c.world = world;
    c.spin_ns = 0;
    c.box = null;
    c.lib.api.ncclAllReduce = Fake.reduce;
    c.lib.api.ncclAllGather = Fake.gather;
    c.api.ncclReduceScatter = Fake.scatter;
    c.api.ncclCommGetAsyncError = Fake.asyncError;
    c.api.ncclCommAbort = Fake.abort;
    return c;
}

fn close(pair: [2]c_int) void {
    for (pair) |fd| if (fd >= 0) {
        _ = std.c.close(fd);
    };
}

test "rounds and windows keep the all-reduce; a wider prompt pass splits, at any --tp" {
    for ([_]usize{ 2, 3, 4, 8 }) |w| {
        for (1..mtp.round_rows + 1) |n| {
            try expect(!exchange.rowSplit(w, n));
            try expectEqual(n, exchange.clearedRows(w, n));
        }
        for ([_]usize{ mtp.round_rows + 1, batch.default_chunk, chunk.lone_rows, batch.max_chunk }) |n| {
            try expect(exchange.rowSplit(w, n));
            try expect(!exchange.rowSplit(1, n));
            try expectEqual(n, exchange.clearedRows(1, n));
        }
    }
}

test "the ranks' rows cover a split pass once, in equal blocks whose pad stays inside the scratch's spare rows" {
    for ([_]usize{ 2, 3, 4, 8 }) |w| {
        var n: usize = mtp.round_rows + 1;
        while (n <= batch.max_chunk) : (n += if (n < 600) 1 else 997) {
            const h = exchange.block(w, n);
            var next: usize = 0;
            for (0..w) |r| {
                const own = exchange.rows(w, n, r);
                // a rank's rows start where its reduce-scatter block does, or it has none
                try expect(own[0] == next and own[1] >= own[0] and own[1] - own[0] <= h);
                try expect(own[0] == r * h or own[0] == n);
                next = own[1];
            }
            try expectEqual(n, next);
            try expect(exchange.clearedRows(w, n) == w * h and w * h - n < w and w * h - n <= exchange.pad_rows);
        }
    }
    // two ranks: the cut it always was
    var n: usize = mtp.round_rows + 1;
    while (n <= 600) : (n += 1) {
        try expectEqual(cuda.rank_scatter.share(n, 2, 0), exchange.rows(2, n, 0));
        try expectEqual(cuda.rank_scatter.share(n, 2, 1), exchange.rows(2, n, 1));
    }
    try expectEqual(@as(usize, 2 * 8192), exchange.clearedRows(2, batch.max_chunk));
    // four ranks, 66 rows: blocks of 17, the last rank's 15 rows and two rows of pad
    try expectEqual([2]usize{ 51, 66 }, exchange.rows(4, 66, 3));
    try expectEqual(@as(usize, 68), exchange.clearedRows(4, 66));
    // eight ranks, 65 rows: blocks of 9, the last rank's two rows
    try expectEqual([2]usize{ 63, 65 }, exchange.rows(8, 65, 7));
}

test "a split pass reduce-scatters its rank's block in place, sums that block's rows and gathers the sums in place" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Fake.launch;
    const bufs: Bufs = .{};
    for ([_]usize{ 2, 4 }) |w| for (0..w) |rank| for ([_]usize{ 65, 66, 1537, batch.default_chunk, batch.max_chunk }) |n| {
        var pair: [2]c_int = .{ -1, -1 };
        var c = try collective(&pair, rank, w, 1000);
        defer close(pair);
        var pass: Pass = .{ .s = &bufs, .k = .{ .moe_sum = .{ .d = &d, .handle = @ptrFromInt(1) } }, .stream = .{ .d = &d, .handle = null } };
        Fake.reset();
        try exchange.run(&pass, &c, n);
        const h = (n + w - 1) / w;
        const lo = rank * h;
        const rows = @min(n, lo + h) - lo;
        try expectEqual(@as(usize, 3), Fake.len);
        const rs = Fake.calls[0];
        try expect(rs.kind == .scatter and rs.f32 and rs.send == bufs.y and rs.recv == bufs.y + lo * row_y and rs.count == h * slots * dims);
        const k = Fake.calls[1];
        try expect(k.kind == .launch);
        try expectEqual([6]u32{ @intCast(rows), dims / 256, 1, 256, 1, 1 }, k.grid);
        try expectEqual([5]u64{ bufs.y + lo * row_y, bufs.wts + lo * slots * 4, bufs.moe_f32 + lo * dims * 4, dims, slots }, k.args);
        const ag = Fake.calls[2];
        try expect(ag.kind == .gather and ag.f32 and ag.send == bufs.moe_f32 + rank * h * dims * 4 and ag.recv == bufs.moe_f32 and ag.count == h * dims);
        try expect(pass.pending_f32 and !c.failed and Fake.aborts == 0);
    };
}

const Outcome = struct { err: ?anyerror, calls: usize, aborted: bool, refused: bool };

/// sum or sumBlock under one failure: what it returned, what reached NCCL, and whether it left the communicator dead.
fn outcome(comptime block: bool, failure: enum { none, empty, peer, async_error, stuck }) !Outcome {
    var pair: [2]c_int = .{ -1, -1 };
    var c = try collective(&pair, 1, 2, 50);
    defer close(pair);
    var d: cuda.Driver = undefined;
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.reset();
    switch (failure) {
        .peer => {
            _ = std.c.close(pair[1]);
            pair[1] = -1;
        },
        .async_error => Fake.state = 3,
        .stuck => Fake.state = nccl.in_progress,
        else => {},
    }
    const count: usize = if (failure == .empty) 0 else 4099;
    const f = if (block) Collective.sumBlock else Collective.sum;
    var err: ?anyerror = null;
    f(&c, stream, 0x5000_0000, count) catch |e| {
        err = e;
    };
    const calls = Fake.len;
    const refused = if (f(&c, stream, 0x5000_0000, 1)) |_| false else |e| e == error.RankDisconnected;
    return .{ .err = err, .calls = calls, .aborted = c.failed and c.comm == null and Fake.aborts == 1, .refused = refused };
}

test "the reduce-scatter keeps the sum's contract: a dead peer, an asynchronous error or the deadline aborts it for good" {
    const cases = .{ .none, .empty, .peer, .async_error, .stuck };
    inline for (cases) |failure| {
        const sum = try outcome(false, failure);
        const scatter = try outcome(true, failure);
        try expectEqual(sum, scatter);
    }
    try expectEqual(Outcome{ .err = null, .calls = 1, .aborted = false, .refused = false }, try outcome(true, .none));
    try expectEqual(Outcome{ .err = null, .calls = 0, .aborted = false, .refused = false }, try outcome(true, .empty));
    try expectEqual(Outcome{ .err = error.RankDisconnected, .calls = 0, .aborted = true, .refused = true }, try outcome(true, .peer));
    try expectEqual(Outcome{ .err = error.NcclFailed, .calls = 1, .aborted = true, .refused = true }, try outcome(true, .async_error));
    try expectEqual(Outcome{ .err = error.RankTimeout, .calls = 1, .aborted = true, .refused = true }, try outcome(true, .stuck));
}

/// Looks at a collective that never finishes queueing, until its 20 ms deadline: how many, with `spin_ns`.
fn stuckPolls(spin_ns: u64) !usize {
    var pair: [2]c_int = .{ -1, -1 };
    var c = try collective(&pair, 0, 2, 20);
    defer close(pair);
    c.spin_ns = spin_ns;
    var d: cuda.Driver = undefined;
    Fake.reset();
    Fake.state = nccl.in_progress;
    try std.testing.expectError(error.RankTimeout, c.sum(.{ .d = &d, .handle = null }, 0x5000_0000, 4));
    try expect(c.failed and Fake.aborts == 1);
    return Fake.polls;
}

test "a wait sleeps between looks unless told to spin first, and meets its deadline either way" {
    const sleeping = try stuckPolls(0);
    const spinning = try stuckPolls(10 * std.time.ns_per_ms);
    // 20 ms of 50 us sleeps is 400 looks at most; half of it spent polling is many times that
    try expect(sleeping >= 1 and sleeping <= 400);
    try expect(spinning > 4 * sleeping);
}

const Box = @import("cuda_box.zig").Box;
const BoxArgs = @import("cuda_box.zig").Args;

/// What cuLaunchKernel was handed for the box's kernel: its one by-value argument.
const BoxFake = struct {
    var args: [8]BoxArgs = undefined;
    var grids: [8]c_uint = undefined;
    var len: usize = 0;

    fn launch(_: abi.Function, gx: c_uint, _: c_uint, _: c_uint, bx: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, raw: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        std.debug.assert(bx == 256);
        args[len] = @as(*const BoxArgs, @ptrCast(@alignCast(raw.?[0].?))).*;
        grids[len] = gx;
        len += 1;
        return 0;
    }
};

/// A box of `world` ranks over a plain page (nothing is mapped to a device; the kernel is the fake's).
fn fakeBox(d: *const cuda.Driver, region: []align(std.heap.page_size_min) u8, rank: u32, world: u32) Box {
    return .{ .d = d, .region = region, .dev = 0x9000_0000, .module = undefined, .func = .{ .d = d, .handle = @ptrFromInt(2) }, .rank = rank, .world = world, .slot_bytes = 262144, .limit = 262144, .timeout_ns = 1_000_000 };
}

test "the box takes a decode row's sum and a small gather as one launch each, numbered in order, and leaves the rest to NCCL" {
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = BoxFake.launch;
    const page = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 4096);
    defer std.testing.allocator.free(page);
    @memset(page, 0);
    var pair: [2]c_int = .{ -1, -1 };
    var c = try collective(&pair, 1, 4, 1000);
    defer close(pair);
    c.box = fakeBox(&d, page, 1, 4);
    const stream: cuda.Stream = .{ .d = &d, .handle = null };
    Fake.reset();
    BoxFake.len = 0;
    // one decode row: 11 slots of 2560 fp32, in place
    try c.sum(stream, 0x1000_0000, 11 * 2560);
    // a split layer's head outputs: 6144 bf16 a rank, this rank's part already in its place of the gathered buffer
    try c.gather(stream, 0x2000_0000 + 1 * 6144 * 2, 0x2000_0000, 6144);
    try c.gatherF32(stream, 0x3000_0000 + 1 * 640 * 4, 0x3000_0000, 640);
    try expectEqual(@as(usize, 3), BoxFake.len);
    try expectEqual(@as(usize, 0), Fake.len); // no NCCL call
    try expectEqual(BoxArgs{ .in = 0x1000_0000, .out = 0x1000_0000, .region = 0x9000_0000, .nbytes = 112640, .slot_bytes = 262144, .timeout_ns = 1_000_000, .rank = 1, .world = 4, .mode = 1, .seq = 1 }, BoxFake.args[0]);
    try expectEqual(@as(c_uint, 7), BoxFake.grids[0]);
    try expectEqual(BoxArgs{ .in = 0x2000_3000, .out = 0x2000_0000, .region = 0x9000_0000, .nbytes = 12288, .slot_bytes = 262144, .timeout_ns = 1_000_000, .rank = 1, .world = 4, .mode = 0, .seq = 2 }, BoxFake.args[1]);
    try expectEqual(@as(c_uint, 1), BoxFake.grids[1]);
    try expect(BoxFake.args[2].nbytes == 2560 and BoxFake.args[2].mode == 0 and BoxFake.args[2].seq == 3);
    // past the limit, and the reduce-scatter at any size: NCCL
    try c.sum(stream, 0x1000_0000, 262144 / 4 + 4);
    try c.sumBlock(stream, 0x1000_0000, 2560);
    try expectEqual(@as(usize, 3), BoxFake.len);
    try expect(Fake.len == 2 and Fake.calls[0].kind == .reduce and Fake.calls[1].kind == .scatter);
    // a size that is not whole 16-byte packs is NCCL's on every rank alike
    try c.gather(stream, 0x2000_0000, 0x2000_0000, 6145);
    try expect(BoxFake.len == 3 and Fake.len == 3 and Fake.calls[2].kind == .gather);
    // an address that is not 16-byte aligned is this rank's error, not a quiet change of path
    try std.testing.expectError(error.UnexpectedTensor, c.sum(stream, 0x1000_0004, 2560));
    try expect(c.failed);
}

test "the box's failure record ends the pass, and an abort reaches the shared page" {
    var d: cuda.Driver = undefined;
    const page = try std.testing.allocator.alignedAlloc(u8, .fromByteUnits(std.heap.page_size_min), 4096);
    defer std.testing.allocator.free(page);
    @memset(page, 0);
    var b = fakeBox(&d, page, 0, 2);
    try b.check();
    b.abort();
    try expectEqual(@as(u32, 1), std.mem.readInt(u32, page[0..4], .little));
    try std.testing.expectError(error.RankDisconnected, b.check());
}
