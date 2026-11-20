//! A shared expert keeps full dot products: gather BF16 intermediate rows, then write disjoint output columns.
const wt = @import("cuda_prompt_wt.zig");
const ops = @import("cuda_batch_ops.zig");
const mixed = @import("cuda_mixed_experts.zig");

pub fn run(self: anytype, native: *const mixed.Owned, n: usize) !void {
    if (!native.split_shared and native.rank != 0) return;
    const s = self.s;
    const dims = wt.dims;
    const glue = self.tri.own orelse return error.NativeGlueRequired;
    // split: each rank an equal share of the intermediate channels and of the output columns
    const world = if (native.split_shared) native.world else 1;
    if (wt.moe_w % world != 0 or dims % world != 0) return error.UnexpectedTensor;
    const width = wt.moe_w / world;
    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[0].native.?, s.shared_gate, s.native_part);
    try self.k.native_lane.matmul(self.stream, s.mixed, dims, n, native.shared[1].native.?, s.shared_up, s.native_part);
    try glue.sharedSwiGLU(self.stream, s.shared_gate, s.shared_up, s.shared_act, n, width);
    var input = s.shared_act;
    if (native.split_shared) {
        const c = self.m.collective orelse return error.TensorParallelUnavailable;
        try c.gather(self.stream, input, s.rank_gather, n * width);
        // one row: the ranks' gathered blocks [world, 1, width] are already the row [1, moe_w] the down projection reads
        if (n > 1) for (0..world) |rank| try ops.copyCols(self.k, self.stream, s.rank_gather + rank * n * width * 2, width, s.rank_input + rank * width * 2, wt.moe_w, width, n);
        input = if (n == 1) s.rank_gather else s.rank_input;
    }
    // the whole expert's down goes straight into its slot (cuda_batch_side.down), as on the side stream
    if (!native.split_shared) return @import("cuda_batch_side.zig").down(self, self.stream, input, native, n);
    try self.k.native_lane.matmulF32(self.stream, input, wt.moe_w, n, native.shared[2].native.?, s.shared_out, s.native_part);
    // Every routed slot and the peer's output columns stay at the zeros set before the expert launches.
    // shared_out is [n, dims / world] fp32, moved as 16-bit words into this rank's columns of the slot
    const cols = dims / world;
    try ops.copyCols(self.k, self.stream, s.shared_out, cols * 2, s.y + (wt.slots - 1) * dims * 4 + native.rank * cols * 4, wt.slots * dims * 2, cols * 2, n);
}
