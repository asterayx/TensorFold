//! ardelva: the ranks' small collectives as one kernel a rank (zig/kernels/cuda/ardelva_box.cu), with no NCCL call.
//!
//! Every rank of one host maps one region of pinned host memory; a rank writes its bytes into its own outbox there
//! and reads the other ranks'. A collective is then one launch on the pass's stream. NCCL keeps what does not fit
//! (`limit`), the reduce-scatter and the broadcast.
//!
//! On unless TF_FLASHNEXT_COLLECTIVE_BOX=0 on some rank. A rank that cannot set it up says so to the others and
//! all of them stay on NCCL: which path a collective takes depends on its size alone, the same on every rank.
const std = @import("std");
const cuda = @import("cuda");
const wire = @import("cuda_rank_wire.zig");

const max_blocks = 16;
const threads = 256;
const ctrl_bytes = 4096;
const flag_bytes = 2 * max_blocks * 128;
const ctrl_rank = 16;
const name_bytes = 64;

pub const Mode = enum(u32) { gather = 0, sum_f32 = 1 };

/// ardelva_box.cu's `box::Args`, field for field.
pub const Args = extern struct {
    in: u64,
    out: u64,
    region: u64,
    nbytes: u64,
    slot_bytes: u64,
    timeout_ns: u64,
    rank: u32,
    world: u32,
    mode: u32,
    seq: u32,
};

/// TF_FLASHNEXT_COLLECTIVE_BOX: unset or 1 asks for the kernel path, 0 keeps every collective on NCCL.
/// On xzss at --tp 2 with the dense split it took a decode forward from 14.5 to 11.8 ms, with the same tokens
/// (asterayx/TensorFold#6).
pub fn setting(value: ?[]const u8) !bool {
    const v = value orelse return true;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadCollectiveBox;
}

/// TF_FLASHNEXT_COLLECTIVE_BOX_KB: the most KiB a rank that a collective sends this way, 4 to 1024; unset: 256.
/// Measured on xzss with a probe of the same exchange (asterayx/TensorFold#1): through host memory a megabyte a rank
/// was slower than NCCL's kernel, and 160 KiB on four cards about even with NCCL's kernel and host call together.
pub fn limitBytes(value: ?[]const u8) !usize {
    const v = value orelse return 256 * 1024;
    const kb = std.fmt.parseInt(usize, v, 10) catch return error.BadCollectiveBox;
    if (kb < 4 or kb > 1024) return error.BadCollectiveBox;
    return kb * 1024;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// The shared mapping's size: ctrl, then every rank's flags and two slots.
pub fn regionBytes(world: usize, slot: usize) usize {
    return ctrl_bytes + world * (flag_bytes + 2 * slot);
}

/// Blocks for `n` bytes a rank: one a 16 KiB, at most `max_blocks`; every rank picks the same for the same bytes.
pub fn blocks(n: usize) u32 {
    return @intCast(@min(max_blocks, @max(1, (n + 16 * 1024 - 1) / (16 * 1024))));
}

pub const Box = struct {
    d: *const cuda.Driver,
    region: []align(std.heap.page_size_min) u8,
    dev: u64,
    module: cuda.Module,
    func: cuda.Function,
    rank: u32,
    world: u32,
    slot_bytes: usize,
    /// The most bytes a rank that go this way (at most a slot).
    limit: usize,
    timeout_ns: u64,
    /// The last exchange's number: the next launch's is one more, on every rank.
    seq: u32 = 0,
    /// The stream of the first launch: the slots' reuse rests on one stream's order.
    stream: ?cuda.abi.Stream = null,

    /// Every rank calls this at the same point after the hello. Null: this group stays on NCCL (the setting is off
    /// somewhere, or a rank could not map the region or load the kernel); every rank then gets null.
    pub fn open(d: *const cuda.Driver, link: wire.Link, rank: usize, world: usize, timeout_ms: u64) !?Box {
        const want = try setting(env("TF_FLASHNEXT_COLLECTIVE_BOX")) and cuda.kernels.ardelva_box.len != 0 and
            !@import("cuda_round_graph.zig").enabled(); // a captured launch would replay one exchange's number
        const limit = try limitBytes(env("TF_FLASHNEXT_COLLECTIVE_BOX_KB"));
        const slot = std.mem.alignForward(usize, limit, 4096);
        const len = regionBytes(world, slot);
        const until = wire.deadline(timeout_ms);
        var msg: [name_bytes]u8 = @splat(0);
        var fd: std.c.fd_t = -1;
        defer if (fd >= 0) {
            _ = std.c.close(fd);
        };
        // rank 0 creates and sizes the file before anyone learns its name; a first byte of 0 says there is none
        if (rank == 0) {
            if (want) made: {
                const s = std.fmt.bufPrint(msg[1 .. name_bytes - 1], "/dev/shm/tf-flashnext-box-{d}-{x}", .{ std.c.getpid(), wire.now() }) catch break :made;
                const path: [*:0]const u8 = @ptrCast(s.ptr); // the buffer is zeroed past the name
                fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
                if (fd >= 0 and std.c.ftruncate(fd, @intCast(len)) == 0) msg[0] = 1;
            }
            try link.send(&msg, until);
        } else {
            try link.receive(&msg, until);
            if (want and msg[0] == 1 and msg[name_bytes - 1] == 0) {
                fd = std.c.open(@ptrCast(msg[1..].ptr), .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            }
        }
        var box: ?Box = if (want and msg[0] == 1 and fd >= 0) map(d, fd, len, rank, world, slot, limit, timeout_ms) catch null else null;
        errdefer if (box) |*b| b.close();
        // every rank's answer, then rank 0's verdict for all
        var go: [1]u8 = .{@intFromBool(box != null)};
        if (rank == 0) {
            for (0..link.peers()) |i| {
                var ok: [1]u8 = .{0};
                try link.peer(i).receive(&ok, until);
                if (ok[0] != 1) go[0] = 0;
            }
            if (msg[0] == 1) _ = std.c.unlink(@ptrCast(msg[1..].ptr));
            try link.send(&go, until);
        } else {
            try link.send(&go, until);
            try link.receive(&go, until);
        }
        if (go[0] != 1) {
            if (box) |*b| b.close();
            if (want and rank == 0) std.log.warn("flashnext: not every rank could set up the collectives' kernel path (or one has TF_FLASHNEXT_COLLECTIVE_BOX=0); every collective stays on NCCL", .{});
            return null;
        }
        if (rank == 0) std.log.info("flashnext: collectives of at most {d} KiB a rank go through one kernel a rank, the rest through NCCL", .{limit / 1024});
        return box;
    }

    fn map(d: *const cuda.Driver, fd: std.c.fd_t, len: usize, rank: usize, world: usize, slot: usize, limit: usize, timeout_ms: u64) !Box {
        const region = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        errdefer std.posix.munmap(region);
        try d.check(d.api.cuMemHostRegister_v2(region.ptr, len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        errdefer _ = d.api.cuMemHostUnregister(region.ptr);
        var dev: u64 = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, region.ptr, 0), "cuMemHostGetDevicePointer");
        var module = try cuda.Module.load(d, cuda.kernels.ardelva_box);
        errdefer module.unload();
        const func = try module.function("ardelva_box");
        return .{ .d = d, .region = region, .dev = dev, .module = module, .func = func, .rank = @intCast(rank), .world = @intCast(world), .slot_bytes = slot, .limit = limit, .timeout_ns = timeout_ms * std.time.ns_per_ms };
    }

    pub fn close(b: *Box) void {
        b.module.unload();
        _ = b.d.api.cuMemHostUnregister(b.region.ptr);
        std.posix.munmap(b.region);
    }

    /// Whether `nbytes` a rank go this way: by the size alone, so every rank answers the same.
    pub fn takes(b: *const Box, nbytes: usize) bool {
        return nbytes != 0 and nbytes <= b.limit and nbytes % 16 == 0;
    }

    /// One exchange: `in`'s `nbytes` out, and into `out` every rank's bytes in rank order (`gather`: [world][nbytes])
    /// or their fp32 sum added in rank order (`sum_f32`: [nbytes], which may be `in`). Queued on `stream`.
    pub fn run(b: *Box, mode: Mode, stream: cuda.Stream, in: u64, out: u64, nbytes: usize) !void {
        // a size that goes this way on one rank does on all; an address or stream that cannot is this rank's error
        if (!b.takes(nbytes) or in % 16 != 0 or out % 16 != 0) return error.UnexpectedTensor;
        if (b.stream) |first| {
            if (first != stream.handle) return error.UnsupportedRankOperation;
        } else b.stream = stream.handle;
        b.seq +%= 1;
        var a: cuda.Args = .{};
        a.add(args(b, mode, in, out, nbytes));
        try cuda.launch.launch(b.func, .{ .grid = .{ .x = blocks(nbytes) }, .block = .{ .x = threads } }, stream, &a);
    }

    fn args(b: *const Box, mode: Mode, in: u64, out: u64, nbytes: usize) Args {
        return .{ .in = in, .out = out, .region = b.dev, .nbytes = nbytes, .slot_bytes = b.slot_bytes, .timeout_ns = b.timeout_ns, .rank = b.rank, .world = b.world, .mode = @backingInt(mode), .seq = b.seq };
    }

    fn ctrl(b: *const Box, i: usize) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(b.region[4 * i ..].ptr));
    }

    /// Ends pending and later exchanges on every rank.
    pub fn abort(b: *Box) void {
        b.ctrl(0).store(1, .release);
    }

    /// An exchange that timed out or was aborted on any rank: every later one did nothing, so the pass's bytes are
    /// not to be used. Asked once the stream is idle.
    pub fn check(b: *const Box) !void {
        if (b.ctrl(0).load(.acquire) == 0) return;
        for (0..b.world) |r| {
            const rec = ctrl_rank + 8 * r;
            if (b.ctrl(rec).load(.acquire) == 0) continue;
            std.log.err("flashnext: rank {d}'s wait for rank {d} in exchange {d} {s} (block {d})", .{ r, b.ctrl(rec + 2).load(.acquire), b.ctrl(rec + 1).load(.acquire), if (b.ctrl(rec + 4).load(.acquire) == 1) "was aborted" else "timed out", b.ctrl(rec + 3).load(.acquire) });
        }
        return error.RankDisconnected;
    }
};

test "the box settings: on unless 0, a limit of 4 to 1024 KiB" {
    try std.testing.expect(try setting(null));
    try std.testing.expect(!try setting("0"));
    try std.testing.expect(try setting("1"));
    try std.testing.expectError(error.BadCollectiveBox, setting("yes"));
    try std.testing.expectEqual(@as(usize, 256 * 1024), try limitBytes(null));
    try std.testing.expectEqual(@as(usize, 64 * 1024), try limitBytes("64"));
    try std.testing.expectError(error.BadCollectiveBox, limitBytes("2"));
    try std.testing.expectError(error.BadCollectiveBox, limitBytes("2048"));
}

test "the box's kernel arguments, region and grid" {
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(Args));
    try std.testing.expectEqual(@as(usize, 4096), flag_bytes);
    try std.testing.expectEqual(@as(usize, 4096 + 4 * (4096 + 2 * 262144)), regionBytes(4, 262144));
    // eight ranks' failure records end inside the ctrl page
    try std.testing.expect(4 * (ctrl_rank + 8 * wire.max_ranks) <= ctrl_bytes);
    try std.testing.expectEqual(@as(u32, 1), blocks(16));
    try std.testing.expectEqual(@as(u32, 1), blocks(16384));
    try std.testing.expectEqual(@as(u32, 2), blocks(16385));
    try std.testing.expectEqual(@as(u32, 7), blocks(11 * 2560 * 4));
    try std.testing.expectEqual(@as(u32, 16), blocks(1 << 20));
}
