//! The shared batched head and last-row write-back, split without changing launch arguments.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const HcKeep = batch.HcKeep;
const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const vocab = wt.vocab;
const pss_n = batch.pss_n;
pub fn head(self: anytype, logits_host: ?[]u8) !void {
    const s = self.s;
    const n = self.rows;
    const mixer_keep: HcKeep = if (self.hooks.keep) |kp| kp.mixer else .{};
    switch (self.hooks.logits) {
        .none => {
            if (self.hooks.streams == 0 and mixer_keep.h == 0) return;
            try self.writeback(0, n);
            try self.copy(self.hooks.streams, s.h, n * wide * 2);
            if (mixer_keep.h != 0) try self.readoutKeep(self.m.mixer, 0, n, mixer_keep);
            return;
        },
        .all => {
            try self.writeback(0, n);
            try self.copy(self.hooks.streams, s.h, n * wide * 2);
            var row0: usize = 0;
            for (self.lanes) |l| {
                try self.copy(l.streams, s.h + row0 * wide * 2, l.rows * wide * 2);
                row0 += l.rows;
            }
            try self.readoutKeep(self.m.mixer, 0, n, mixer_keep);
            const out = if (self.hooks.logits_out != 0) self.hooks.logits_out else return error.UnexpectedTensor;
            try self.projectHead(out, n);
            row0 = 0;
            for (self.lanes) |l| {
                try self.copy(l.logits, out + row0 * vocab * 2, l.rows * vocab * 2);
                row0 += l.rows;
            }
            if (logits_host) |lh| {
                try self.complete();
                const d = self.stream.d;
                try cuda.memory.download(d, out + (n - 1) * vocab * 2, std.mem.sliceAsBytes(lh)[0 .. vocab * 2]);
            }
        },
        .last => {
            // the last row's streams, sums, gates and MoE slots moved to row 0, which keeps 16-byte alignment
            const r = n - 1;
            if (self.hooks.streams != 0) {
                // every row written back and kept, then the last row's streams and sums at row 0
                try self.writeback(0, n);
                try self.copy(self.hooks.streams, s.h, n * wide * 2);
                if (r > 0) {
                    try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                    try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                }
            } else if (r > 0) {
                try self.copy(s.h, s.h + r * wide * 2, wide * 2);
                try self.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
                try self.copy(s.inj, s.inj + r * streams * 2, streams * 2);
                try self.copy(s.y, s.y + r * slots * dims * 4, slots * dims * 4);
                try self.copy(s.wts, s.wts + r * slots * 4, slots * 4);
                if (self.pending_f32) try self.copy(s.moe_f32, s.moe_f32 + r * dims * 4, dims * 4);
            }
            if (self.hooks.streams == 0) try self.writeback(0, 1);
            try self.readout(self.m.mixer, 0, 1);
            try self.projectHead(s.logits, 1);
            if (logits_host) |lh| {
                try self.complete();
                const d = self.stream.d;
                try cuda.memory.download(d, s.logits, std.mem.sliceAsBytes(lh)[0 .. vocab * 2]);
            }
        },
    }
}

/// Column tiles keep their own reduction order; each gathered row restores global vocabulary order.
pub fn projectHead(self: anytype, out: u64, n: usize) !void {
    const s = self.s;
    if (self.m.head.n == vocab) {
        _ = try self.lane(s.mixed, s.xsm, self.m.head, out, s.dummy, n, 1);
        return;
    }
    const c = self.m.collective orelse return error.TensorParallelUnavailable;
    const p = try @import("cuda_head.zig").part(vocab, c.world, c.rank);
    if (self.m.head.n != p.rows or n > s.cap or n > s.head_rows) return error.UnexpectedTensor;
    _ = try self.lane(s.mixed, s.xsm, self.m.head, s.head_local, s.dummy, n, 1);
    if (p.exact(c.world, vocab)) {
        for (0..n) |r| try c.gather(self.stream, s.head_local + r * p.rows * 2, out + r * vocab * 2, p.rows);
        return;
    }
    // uneven shards: every rank sends a whole block (the last rank's ends in bytes past its rows, which land past
    // the row's last id), into a row of blocks kept after the local columns, whose first `vocab` ids are the row
    const local = n * p.rows * 2;
    const row = std.mem.alignForward(usize, local + (p.block - p.rows) * 2, 4096);
    if (row + c.world * p.block * 2 > s.head_rows * vocab) return error.UnexpectedTensor;
    for (0..n) |r| {
        try c.gather(self.stream, s.head_local + r * p.rows * 2, s.head_local + row, p.block);
        try self.copy(out + r * vocab * 2, s.head_local + row, vocab * 2);
    }
}

/// A pass that records what the split head asks of the device and of the ranks.
const FakeHead = struct {
    const Call = struct { kind: enum { lane, gather, copy }, src: u64, dst: u64, n: usize };
    const Ranks = struct {
        world: usize,
        rank: usize,
        log: *std.ArrayList(Call),
        pub fn gather(c: *Ranks, _: void, local: u64, out: u64, count: usize) !void {
            c.log.appendAssumeCapacity(.{ .kind = .gather, .src = local, .dst = out, .n = count });
        }
    };
    m: struct { head: struct { n: usize }, collective: ?*Ranks },
    s: *const struct { cap: usize = 64, head_rows: usize = 64, head_local: u64 = 1 << 32, mixed: u64 = 0, xsm: u64 = 0, dummy: u64 = 0 },
    stream: void = {},
    log: *std.ArrayList(Call),
    pub fn lane(self: *FakeHead, _: u64, _: u64, proj: anytype, out: u64, _: u64, n: usize, _: usize) !usize {
        self.log.appendAssumeCapacity(.{ .kind = .lane, .src = proj.n, .dst = out, .n = n });
        return 0;
    }
    pub fn copy(self: *FakeHead, dst: u64, src: u64, n: usize) !void {
        self.log.appendAssumeCapacity(.{ .kind = .copy, .src = src, .dst = dst, .n = n });
    }
};

test "the split head at four ranks gathers whole blocks beside the local columns and copies each row's ids out" {
    const part = @import("cuda_head.zig").part;
    var buf: [32]FakeHead.Call = undefined;
    for ([_]usize{ 0, 3 }) |rank| {
        var log: std.ArrayList(FakeHead.Call) = .initBuffer(&buf);
        const p = try part(vocab, 4, rank);
        var ranks: FakeHead.Ranks = .{ .world = 4, .rank = rank, .log = &log };
        var f: FakeHead = .{ .m = .{ .head = .{ .n = p.rows }, .collective = &ranks }, .s = &.{}, .log = &log };
        const out: u64 = 1 << 40;
        // eight rows: the last rank's columns then end on a page, so only the margin keeps the row of blocks clear
        try projectHead(&f, out, 8);
        const base = f.s.head_local;
        try std.testing.expectEqual(@as(usize, 17), log.items.len);
        try std.testing.expect(log.items[0].kind == .lane and log.items[0].src == p.rows and log.items[0].dst == base and log.items[0].n == 8);
        const row = log.items[1].dst;
        // the row of blocks starts after everything a rank sends from: eight rows of columns and the last rank's
        // block running past its rows
        try std.testing.expect(row >= base + 7 * p.rows * 2 + p.block * 2 and row % 16 == 0);
        try std.testing.expect(row + 4 * p.block * 2 <= base + f.s.head_rows * vocab);
        for (0..8) |r| {
            const g = log.items[1 + 2 * r];
            const c = log.items[2 + 2 * r];
            try std.testing.expect(g.kind == .gather and g.src == base + r * p.rows * 2 and g.dst == row and g.n == p.block);
            try std.testing.expect(c.kind == .copy and c.src == row and c.dst == out + r * vocab * 2 and c.n == vocab * 2);
        }
        // a rank whose loaded projection is another rank's share is refused
        f.m.head.n = if (rank == 3) p.block else p.rows - 512;
        try std.testing.expectError(error.UnexpectedTensor, projectHead(&f, out, 3));
    }
}

test "the split head at two ranks still gathers each row's halves in place" {
    var buf: [8]FakeHead.Call = undefined;
    var log: std.ArrayList(FakeHead.Call) = .initBuffer(&buf);
    var ranks: FakeHead.Ranks = .{ .world = 2, .rank = 1, .log = &log };
    var f: FakeHead = .{ .m = .{ .head = .{ .n = vocab / 2 }, .collective = &ranks }, .s = &.{}, .log = &log };
    try projectHead(&f, 1 << 40, 2);
    try std.testing.expectEqual(@as(usize, 3), log.items.len);
    for (0..2) |r| {
        const g = log.items[1 + r];
        try std.testing.expect(g.kind == .gather and g.src == f.s.head_local + r * vocab and g.dst == (1 << 40) + r * vocab * 2 and g.n == vocab / 2);
    }
}
