// The outbox kernel (mailbox.cu includes it; outbox.zig is its host side): tp_mailbox's exchange for 2 to 8 ranks
// of one host. A rank writes its shard once, into its own outbox in pinned host memory every GPU maps, and reads
// the other ranks' outboxes; tp_mailbox writes into the peer's inbox, which is the same thing at two ranks.
//
// Why an outbox and not an inbox a peer: on four RTX PRO 5000 (two a CPU) a kernel that wrote into three peers at
// once was measured many times slower than one write, and one write into host memory was the fastest of the forms
// tried for shards up to about 10 KiB (asterayx/TensorFold#1). Nothing here was measured on other machines.

namespace tftp {

constexpr int kOutMaxRanks = 8;

struct OutArgs {
    const uint8_t *in;     // this rank's shard (device)
    uint8_t *out;          // gather [world n], sums [n]
    uint8_t *region;       // the shared mapping as this GPU maps it: ctrl, then an outbox a rank
    uint32_t *state;       // device: [0] epoch, [1] tail counter, [2] poison
    uint64_t nbytes;       // bytes a rank
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank;
    uint32_t world;
    uint32_t mode;         // kGather, or at two ranks kSumF32 / kSumBf16
    uint32_t pad;
};

constexpr uint64_t kOutCtrlBytes = 4096;
constexpr uint64_t kOutFlagBytes = 2 * kMaxBlocks * kFlagStride * 4;  // [2 slots][kMaxBlocks] flags, a line each

// Rank r's outbox: its flags, then its two slots.
__device__ __forceinline__ uint8_t *outbox_of(const OutArgs &a, uint32_t r) {
    return a.region + kOutCtrlBytes + static_cast<uint64_t>(r) * (kOutFlagBytes + 2u * a.slot_bytes);
}

}  // namespace tftp

extern "C" __global__ void __launch_bounds__(512) tp_outbox(tftp::OutArgs a) {
    using namespace tftp;
    __shared__ uint32_t s_head[2];  // poison, seq
    __shared__ uint32_t s_bad;
    const uint32_t tid = threadIdx.x;
    const uint32_t gdim = gridDim.x;
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + tid;
    const uint64_t stride = static_cast<uint64_t>(gdim) * blockDim.x;
    uint32_t *ctrl = reinterpret_cast<uint32_t *>(a.region);

    // as tp_mailbox: a poisoned transport does nothing, and the epoch names this exchange on every rank
    if (tid == 0) {
        s_head[0] = ld_relaxed_gpu(a.state + 2);
        s_head[1] = ld_relaxed_gpu(a.state) + 1u;
        s_bad = 0u;
    }
    __syncthreads();
    if (s_head[0] != 0u) {
        return;
    }
    const uint32_t seq = s_head[1];
    const uint64_t slot = seq & 1u;
    const uint64_t flag_at = (slot * kMaxBlocks + blockIdx.x) * kFlagStride;

    // 1. this block's share of the shard into this rank's outbox, then the block's flag after it. The slot of this
    //    parity was last read for seq - 2, and every rank finished that before it flagged seq - 1, which this
    //    rank's seq - 1 waited for.
    uint8_t *mine = outbox_of(a, a.rank);
    copy_bytes<false>(mine + kOutFlagBytes + slot * a.slot_bytes, a.in, a.nbytes, index, stride);
    __syncthreads();
    if (tid == 0) {
        fence_sc_sys();
        st_release_sys(reinterpret_cast<uint32_t *>(mine) + flag_at, seq);
    }
    // the local shard while the other ranks' writes travel
    if (a.mode == kGather) {
        copy_bytes<false>(a.out + static_cast<uint64_t>(a.rank) * a.nbytes, a.in, a.nbytes, index, stride);
    }

    // 2. every other rank's block of the same index (it wrote exactly the bytes this block reads)
    if (tid == 0) {
        for (uint32_t r = 0; r < a.world && s_bad == 0u; ++r) {
            if (r == a.rank) continue;
            uint32_t seen = 0u, why = 0u, waited = 0u;
            if (!wait_flag(reinterpret_cast<const uint32_t *>(outbox_of(a, r)) + flag_at, seq, ctrl, a.timeout_ns, &seen,
                           &why, &waited)) {
                s_bad = 1u;
                // the record's block word also names the rank waited for: block | rank << 8
                fail_record(a.state + 2, ctrl + kCtrlRank + 8u * a.rank, seq, seen, blockIdx.x | (r << 8), why, waited);
            }
        }
    }
    __syncthreads();

    // 3. copy out, gated by this block's own waits
    if (s_bad == 0u) {
        if (a.mode == kGather) {
            for (uint32_t r = 0; r < a.world; ++r) {
                if (r == a.rank) continue;
                copy_bytes<true>(a.out + static_cast<uint64_t>(r) * a.nbytes,
                                 outbox_of(a, r) + kOutFlagBytes + slot * a.slot_bytes, a.nbytes, index, stride);
            }
        } else {
            // two ranks only (outbox.zig refuses the rest): tp_mailbox's sum, bit for bit
            const uint8_t *peer = outbox_of(a, 1u - a.rank) + kOutFlagBytes + slot * a.slot_bytes;
            if (a.mode == kSumF32) {
                sum_bytes<false>(a.out, a.in, peer, a.nbytes, a.rank, index, stride);
            } else {
                sum_bytes<true>(a.out, a.in, peer, a.nbytes, a.rank, index, stride);
            }
        }
    }

    // 4. the last block publishes the epoch, unless a wait of this launch failed (later launches stay no-ops)
    __syncthreads();
    if (tid == 0) {
        const bool last = gdim == 1u || atom_add_acq_rel_gpu(a.state + 1, 1u) == gdim - 1u;
        const bool ok = gdim == 1u ? s_bad == 0u : ld_relaxed_gpu(a.state + 2) == 0u;
        if (last && gdim > 1u) {
            st_release_gpu(a.state + 1, 0u);
        }
        if (last && ok) {
            st_release_gpu(a.state, seq);
        }
    }
}
