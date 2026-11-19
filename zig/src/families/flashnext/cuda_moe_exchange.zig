//! A prompt pass's routed-expert exchange at --tp: each rank sums its block of rows' slots, then gathers the sums.
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const mtp = @import("cuda_mtp.zig");
const moe = @import("cuda_batch_moe.zig");

const dims = wt.dims;
const slots = wt.slots;

/// Rows the scratch keeps past a pass's own for the exchange's last blocks: a row a rank after the first at most.
pub const pad_rows = @import("cuda_rank_wire.zig").max_ranks - 1;

/// Rows past any decode round or verify window: `world` and `n` are what every rank's pass agrees on.
pub fn rowSplit(world: usize, n: usize) bool {
    return world > 1 and n > mtp.round_rows;
}

/// Rows of each rank's reduce-scatter block: the rows over the ranks, rounded up (the last blocks pad to it).
pub fn block(world: usize, n: usize) usize {
    return (n + world - 1) / world;
}

/// Rank `rank`'s rows [lo, hi) of a split pass: its block's, cut at the pass's end (none past it).
pub fn rows(world: usize, n: usize, rank: usize) [2]usize {
    const h = block(world, n);
    return .{ @min(n, rank * h), @min(n, (rank + 1) * h) };
}

/// Rows of y the clear spans: a split pass's blocks, else the pass's own.
pub fn clearedRows(world: usize, n: usize) usize {
    return if (rowSplit(world, n)) world * block(world, n) else n;
}

/// y's slots, a rank's values and every other rank's +0, to their fp32 sum in moe_f32 on every rank: the pending_f32
/// form. A slot's row is another value on one rank alone, so the order the ranks are added in cannot change a bit.
pub fn run(pass: anytype, c: anytype, n: usize) !void {
    const s = pass.s;
    const h = block(c.world, n);
    const own = rows(c.world, n, c.rank);
    try c.sumBlock(pass.stream, s.y, h * slots * dims);
    if (own[1] > own[0]) try moe.sumSlots(pass, own[0], own[1] - own[0]);
    try c.gatherF32(pass.stream, s.moe_f32 + c.rank * h * dims * 4, s.moe_f32, h * dims);
    pass.pending_f32 = true;
}

test {
    if (@import("builtin").os.tag != .windows) _ = @import("cuda_moe_exchange_test.zig"); // over a Unix socket pair
}
