//! The outbox backend: the mailbox's one-kernel exchange for 2 to 8 ranks of one host (host side of outbox.cuh).
//!
//! A rank writes its shard once, into its own outbox in pinned host memory every GPU maps, and reads the other
//! ranks' outboxes. All-gathers at any world; fp32 and bf16 sums at two ranks only, where they are the mailbox's
//! bits. Past two ranks a sum stays on NCCL, as it does on the multirank RoCE backend: the suite holds a
//! multirank sum to NCCL's bytes, and NCCL adds in its ring's order.
const std = @import("std");
const log = @import("log.zig");
const cuda = @import("cuda");
const collective = @import("collective.zig");
const mailbox = @import("mailbox.zig");
const Bootstrap = @import("bootstrap.zig").Bootstrap;
const sock = @import("sock.zig");
const posix = std.posix;
const DevicePtr = collective.DevicePtr;
const DType = collective.DType;
const Error = collective.Error;

pub const max_ranks = 8;
const threads = 512;
const flag_bytes = 2 * mailbox.max_blocks * 128;
const ctrl_bytes = 4096;
const ctrl_abort = 0;
const ctrl_rank = 16;

const Mode = enum(u32) { gather = 0, sum_f32 = 1, sum_bf16 = 2 };

/// outbox.cuh's `OutArgs`, field for field.
const OutArgs = extern struct {
    in: DevicePtr,
    out: DevicePtr,
    region: DevicePtr,
    state: DevicePtr,
    nbytes: u64,
    slot_bytes: u64,
    timeout_ns: u64,
    rank: u32,
    world: u32,
    mode: u32,
    pad: u32 = 0,
};

/// The shared mapping's size for `world` ranks with slots of `slot` bytes: ctrl, then flags and two slots a rank.
pub fn regionBytes(world: u32, slot: usize) usize {
    return ctrl_bytes + @as(usize, world) * (flag_bytes + 2 * slot);
}

pub const Outbox = struct {
    d: *const cuda.Driver,
    me: u32,
    world: u32,
    slot_bytes: usize,
    timeout_ns: u64,
    region: []align(std.heap.page_size_min) u8,
    dev: DevicePtr,
    hr: mailbox.HostRegister,
    state: cuda.DeviceBuffer,
    module: cuda.Module,
    func: cuda.Function,
    aborted: std.atomic.Value(bool) = .init(false),

    /// Every rank calls it after the bootstrap (2 to 8 ranks, one host); `image` is mailbox.cu's fatbin.
    pub fn init(d: *const cuda.Driver, b: *Bootstrap, image: []const u8, max_bytes: usize, timeout_ns: u64) !Outbox {
        if (b.world < 2 or b.world > max_ranks) return error.Unsupported;
        if (image.len == 0) return error.NoKernel;
        const slot = std.mem.alignForward(usize, max_bytes, 4096);
        const len = regionBytes(b.world, slot);
        // as the mailbox: rank 0 creates and sizes the file before anyone learns its name, the others open it,
        // then rank 0 unlinks it
        var name: [64]u8 = undefined;
        var name_len: u8 = 0;
        var fd: std.c.fd_t = -1;
        defer if (fd >= 0) {
            _ = std.c.close(fd);
        };
        if (b.rank == 0) blk: {
            const s = std.fmt.bufPrint(&name, "/dev/shm/tf-tp-out-{d}-{x}", .{ std.c.getpid(), sock.nowNs() }) catch break :blk;
            name[s.len] = 0;
            fd = std.c.open(name[0..s.len :0], .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
            if (fd >= 0 and std.c.ftruncate(fd, @intCast(len)) == 0) name_len = @intCast(s.len);
        }
        // an empty name tells the other ranks that rank 0 failed (they must not wait for a name that never comes)
        try b.broadcast(std.mem.asBytes(&name_len), std.mem.asBytes(&name_len));
        if (name_len == 0) return error.SharedMemoryUnavailable;
        try b.broadcast(name[0..name_len], name[0..name_len]);
        name[name_len] = 0;
        const path = name[0..name_len :0];
        if (b.rank != 0) {
            fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CLOEXEC = true }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.SharedMemoryUnavailable;
        }
        try b.agree("mapped", "outbox file");
        const region = try posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        errdefer posix.munmap(region);
        try b.agree("open", "outbox mapping");
        if (b.rank == 0) _ = std.c.unlink(path);
        var hr = try mailbox.HostRegister.open();
        errdefer hr.lib.close();
        try d.check(hr.register(region.ptr, len, 1 | 2), "cuMemHostRegister"); // PORTABLE | DEVICEMAP
        errdefer _ = hr.unregister(region.ptr);
        var dev: DevicePtr = 0;
        try d.check(d.api.cuMemHostGetDevicePointer_v2(&dev, region.ptr, 0), "cuMemHostGetDevicePointer");
        var state = try cuda.DeviceBuffer.alloc(d, 16);
        errdefer state.free();
        try state.fill8(0, null);
        var module = try cuda.Module.load(d, image);
        errdefer module.unload();
        // an image built before the outbox existed has no such kernel: the session falls back to NCCL on every rank
        const func = module.function("tp_outbox") catch return error.NoKernel;
        try b.agree("ready", "outbox setup");
        return .{ .d = d, .me = b.rank, .world = b.world, .slot_bytes = slot, .timeout_ns = timeout_ns, .region = region, .dev = dev, .hr = hr, .state = state, .module = module, .func = func };
    }

    pub fn deinit(m: *Outbox) void {
        m.module.unload();
        m.state.free();
        _ = m.hr.unregister(m.region.ptr);
        m.hr.lib.close();
        posix.munmap(m.region);
    }

    fn ctrlWord(m: *const Outbox, i: usize) *std.atomic.Value(u32) {
        return @ptrCast(@alignCast(m.region[4 * i ..].ptr));
    }

    /// Whether `n` bytes a rank fit one slot.
    pub fn fits(m: *const Outbox, n: usize) bool {
        return n <= m.slot_bytes;
    }

    fn launch(m: *Outbox, mode: Mode, in: DevicePtr, out: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        if (m.aborted.load(.acquire)) return error.Aborted;
        if (n > m.slot_bytes) return error.Unsupported;
        if (n == 0) return;
        var a: OutArgs = .{
            .in = in,
            .out = out,
            .region = m.dev,
            .state = m.state.ptr,
            .nbytes = n,
            .slot_bytes = m.slot_bytes,
            .timeout_ns = m.timeout_ns,
            .rank = m.me,
            .world = m.world,
            .mode = @backingInt(mode),
        };
        var params = [1]?*anyopaque{@ptrCast(&a)};
        const r = m.d.api.cuLaunchKernel(m.func.handle, mailbox.blocksFor(n), 1, 1, threads, 1, 1, 0, stream, &params, null);
        m.d.check(r, "cuLaunchKernel tp_outbox") catch return error.BackendFailed;
    }

    pub fn allGatherBytes(m: *Outbox, s: DevicePtr, r: DevicePtr, n: usize, stream: collective.Stream) Error!void {
        return m.launch(.gather, s, r, n, stream);
    }

    /// Whether this outbox adds: two ranks only (see the file's head).
    pub fn sums(m: *const Outbox) bool {
        return m.world == 2;
    }

    /// fp32 or bf16 at two ranks; the rest is `error.Unsupported` (the route sends it to NCCL before it gets here).
    pub fn sum(m: *Outbox, s: DevicePtr, r: DevicePtr, count: usize, t: DType, stream: collective.Stream) Error!void {
        if (!m.sums()) return error.Unsupported;
        const mode: Mode = switch (t) {
            .f32 => .sum_f32,
            .bf16 => .sum_bf16,
            else => return error.Unsupported,
        };
        return m.launch(mode, s, r, count * t.size(), stream);
    }

    /// The pairwise exchange is not an outbox operation: the route sends it to NCCL.
    pub fn exchangeBytes(_: *Outbox, _: DevicePtr, _: DevicePtr, _: usize, _: collective.Stream) Error!void {
        return error.Unsupported;
    }

    /// Ends pending and later exchanges on every rank (the abort word is shared); this rank's launches stop.
    pub fn abort(m: *Outbox) void {
        m.aborted.store(true, .release);
        m.ctrlWord(ctrl_abort).store(1, .release);
    }

    /// A wait that timed out or was aborted on any rank.
    pub fn check(m: *Outbox) Error!void {
        for (0..m.world) |r| {
            const rec = ctrl_rank + 8 * r;
            if (m.ctrlWord(rec).load(.acquire) != 0) {
                const who = m.ctrlWord(rec + 3).load(.acquire);
                log.warn("tp outbox: rank {d}'s wait for rank {d}'s seq {d} failed ({s} after {d} us; block {d}, flag held {d})", .{ r, who >> 8, m.ctrlWord(rec + 1).load(.acquire), if (m.ctrlWord(rec + 4).load(.acquire) == 1) "aborted" else "timed out", m.ctrlWord(rec + 5).load(.acquire), who & 0xff, m.ctrlWord(rec + 2).load(.acquire) });
                return error.PeerFailed;
            }
        }
        if (m.aborted.load(.acquire) or m.ctrlWord(ctrl_abort).load(.acquire) != 0) return error.Aborted;
    }
};

test "the kernel's argument block and the region's layout" {
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(OutArgs));
    try std.testing.expectEqual(@as(usize, 4096), flag_bytes);
    // ctrl, then four outboxes of flags and two 16 KiB slots
    try std.testing.expectEqual(@as(usize, 4096 + 4 * (4096 + 2 * 16384)), regionBytes(4, 16384));
    // the failure records of eight ranks end inside the ctrl page
    try std.testing.expect(4 * (ctrl_rank + 8 * max_ranks) <= ctrl_bytes);
}

test "the route adds through the outbox at two ranks and through NCCL past them" {
    const hybrid = @import("hybrid.zig");
    var two: Outbox = undefined;
    two.world = 2;
    var four: Outbox = undefined;
    four.world = 4;
    const a: hybrid.Hybrid = .{ .small = .{ .outbox = &two }, .nccl = undefined, .max_bytes = 16384 };
    const b: hybrid.Hybrid = .{ .small = .{ .outbox = &four }, .nccl = undefined, .max_bytes = 16384 };
    try std.testing.expect(a.sumsSmall());
    try std.testing.expect(!b.sumsSmall());
    try std.testing.expectError(error.Unsupported, four.sum(0, 0, 4, .f32, null));
}
