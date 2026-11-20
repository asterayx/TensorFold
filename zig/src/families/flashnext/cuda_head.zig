//! Optional vocabulary-column partitioning: each rank retains the checkpoint bits for its own output rows.
const std = @import("std");
const Tensor = @import("core").safetensors.Tensor;

pub fn setting(value: ?[]const u8) !bool {
    const v = value orelse return false;
    if (std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadHeadSharding;
}

pub fn enabled() !bool {
    return setting(if (std.c.getenv("TF_FLASHNEXT_SHARD_HEAD")) |v| std.mem.span(v) else null);
}

/// The projection's tiles are 256 output rows; a shard is whole tiles.
pub const tile = 256;

/// One rank's rows of a `rows`-row projection shared between `world` ranks.
pub const Part = struct {
    /// The rank's first row and its rows.
    start: usize,
    rows: usize,
    /// The rows every rank's gathered block holds: the tiles over the ranks, rounded up. Every rank but the last
    /// has exactly that many, so the blocks of a gather land each at its rank's first row; the last rank has what
    /// is left, and its block's tail is past the last row.
    block: usize,

    /// Whether the ranks' blocks are the rows and nothing more: a gather can then write the row in place.
    pub fn exact(p: Part, world: usize, rows: usize) bool {
        return world * p.block == rows;
    }
};

pub fn part(rows: usize, world: usize, rank: usize) !Part {
    if (world < 2 or rank >= world or rows == 0 or rows % tile != 0) return error.BadHeadSharding;
    const tiles = rows / tile;
    const each = (tiles + world - 1) / world;
    // the last rank's share has to be something
    if (each * (world - 1) >= tiles) return error.BadHeadSharding;
    const start = rank * each;
    return .{ .start = start * tile, .rows = (if (rank + 1 == world) tiles - start else each) * tile, .block = each * tile };
}

pub fn shard(t: Tensor, world: usize, rank: usize) !Tensor {
    if (t.rank != 2 or t.dtype != .bf16) return error.BadHeadSharding;
    const p = try part(t.dim(0), world, rank);
    const bytes = try std.math.mul(usize, try std.math.mul(usize, t.dim(0), t.dim(1)), 2);
    if (t.bytes.len != bytes) return error.UnexpectedTensor;
    var out = t;
    out.shape[0] = p.rows;
    out.bytes = t.bytes[p.start * t.dim(1) * 2 ..][0 .. p.rows * t.dim(1) * 2];
    return out;
}

test "head partitions preserve every BF16 byte in global vocabulary order" {
    var bytes: [512 * 32 * 2]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i * 17 + i / 251);
    const t: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 512, 32, 0, 0, 0 }, .bytes = &bytes };
    const a = try shard(t, 2, 0);
    const b = try shard(t, 2, 1);
    try std.testing.expectEqual(@as(usize, 256), a.dim(0));
    try std.testing.expectEqualSlices(u8, bytes[0 .. bytes.len / 2], a.bytes);
    try std.testing.expectEqualSlices(u8, bytes[bytes.len / 2 ..], b.bytes);
    try std.testing.expectError(error.BadHeadSharding, shard(t, 1, 0));
    try std.testing.expectError(error.BadHeadSharding, shard(t, 2, 2));
    var bad = t;
    bad.shape[0] = 511;
    try std.testing.expectError(error.BadHeadSharding, shard(bad, 2, 0));
    // one tile is not two ranks' rows
    bad.shape[0] = 256;
    try std.testing.expectError(error.BadHeadSharding, shard(bad, 2, 0));
    bad = t;
    bad.bytes = bytes[1..];
    try std.testing.expectError(error.UnexpectedTensor, shard(bad, 2, 0));
    try std.testing.expect(!try setting(null));
    try std.testing.expect(!try setting("0"));
    try std.testing.expect(try setting("1"));
    try std.testing.expectError(error.BadHeadSharding, setting("2"));
}

test "the vocabulary's 970 tiles between four ranks: 243 each for three and 241 for the last, in order and once" {
    const vocab = 248320;
    var next: usize = 0;
    for (0..4) |rank| {
        const p = try part(vocab, 4, rank);
        try std.testing.expectEqual(next, p.start);
        try std.testing.expectEqual(@as(usize, 243 * 256), p.block);
        try std.testing.expectEqual(@as(usize, if (rank == 3) 241 * 256 else 243 * 256), p.rows);
        // a rank's block of a gather starts at its first row
        try std.testing.expectEqual(rank * p.block, p.start);
        try std.testing.expect(!p.exact(4, vocab));
        next += p.rows;
    }
    try std.testing.expectEqual(@as(usize, vocab), next);
    // two ranks: the halves they always were, with nothing past the last row
    for (0..2) |rank| {
        const p = try part(vocab, 2, rank);
        try std.testing.expect(p.start == rank * (vocab / 2) and p.rows == vocab / 2 and p.exact(2, vocab));
    }
    // more ranks than can each have a tile
    try std.testing.expectError(error.BadHeadSharding, part(3 * 256, 4, 0));
    try std.testing.expectError(error.BadHeadSharding, part(vocab + 1, 4, 0));
}

test "four ranks' shards are the checkpoint's bytes in vocabulary order, uneven where the tiles are" {
    // 7 tiles of 4 inputs: 2, 2, 2 and 1 tiles
    var bytes: [7 * 256 * 4 * 2]u8 = undefined;
    for (&bytes, 0..) |*v, i| v.* = @truncate(i * 31 + i / 253);
    const t: Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 7 * 256, 4, 0, 0, 0 }, .bytes = &bytes };
    var at: usize = 0;
    for (0..4) |rank| {
        const one = try shard(t, 4, rank);
        try std.testing.expectEqual(@as(usize, if (rank == 3) 256 else 512), one.dim(0));
        try std.testing.expectEqualSlices(u8, bytes[at..][0..one.bytes.len], one.bytes);
        at += one.bytes.len;
    }
    try std.testing.expectEqual(bytes.len, at);
}
