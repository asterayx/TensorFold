// ardelva: Flash Next's small collectives between the ranks of one host as one kernel a rank, with no NCCL call.
// The host side is zig/src/families/flashnext/cuda_box.zig.
//
// Why: at --tp a decode forward makes 48 to 192 collectives of a few KB to about 100 KB. NCCL's kernel for one took
// about 9 us of GPU time on xzss, and its host call about 20 us more, during which the GPU has nothing queued
// (asterayx/TensorFold#5). Here a collective is one launch.
//
// How: every rank maps one region of pinned host memory. A rank writes its bytes once, into its own outbox there,
// sets a flag a block after them, waits for the same block's flag of every other rank and reads their outboxes.
// The orderings (16-byte stores, one fence and a release store of the flag after a block's data, an acquire load
// in the wait) are upstream's fabric/tp/mailbox.cu's.
//
// The exchange's number comes from the host, the same on every rank, since every rank makes the same collectives
// in the same order. An outbox has two slots, taken by the number's parity: a rank writes exchange s + 2 into the
// slot of s only after its exchange s + 1 saw every other rank's flag for s + 1, and a rank sets that flag only
// when its kernel for s, which read the slot, is over (one stream a rank, launches in order).
#include <cstdint>

#include <cuda_runtime.h>

namespace box {

constexpr uint32_t kMaxBlocks = 16;
constexpr uint32_t kFlagWords = 32;                                 // a flag to a 128-byte line
constexpr uint64_t kCtrlBytes = 4096;                               // word 0: abort; words 16 + 8 r: rank r's failure
constexpr uint64_t kFlagBytes = 2ull * kMaxBlocks * kFlagWords * 4;  // [2 slots][kMaxBlocks] flags
constexpr uint32_t kCtrlRank = 16;
constexpr uint32_t kGather = 0;  // any other mode: an fp32 sum
constexpr uint32_t kAborted = 1, kTimedOut = 2;

struct Args {
    const uint8_t *in;    // this rank's bytes (device, 16-byte aligned)
    uint8_t *out;         // gather: [world][nbytes]; sum: [nbytes] fp32, which may be `in`
    uint8_t *region;      // the shared mapping as this GPU maps it
    uint64_t nbytes;      // bytes a rank, a multiple of 16
    uint64_t slot_bytes;
    uint64_t timeout_ns;
    uint32_t rank, world, mode, seq;
};

__device__ __forceinline__ uint32_t ld_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint32_t ld_acquire_sys(const uint32_t *p) {
    uint32_t v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void st_release_sys(uint32_t *p, uint32_t v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }
__device__ __forceinline__ uint4 ld_sys_v4(const uint4 *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ uint64_t timer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// Rank r's outbox: its flags, then its two slots.
__device__ __forceinline__ uint8_t *outbox(const Args &a, uint32_t r) {
    return a.region + kCtrlBytes + (uint64_t)r * (kFlagBytes + 2 * a.slot_bytes);
}
__device__ __forceinline__ uint32_t *flag(const Args &a, uint32_t r) {
    return (uint32_t *)outbox(a, r) + ((uint64_t)(a.seq & 1u) * kMaxBlocks + blockIdx.x) * kFlagWords;
}
__device__ __forceinline__ uint4 *slot(const Args &a, uint32_t r) {
    return (uint4 *)(outbox(a, r) + kFlagBytes + (uint64_t)(a.seq & 1u) * a.slot_bytes);
}
__device__ __forceinline__ uint4 add_f32(uint4 x, uint4 y) {
    uint4 z;
    z.x = __float_as_uint(__fadd_rn(__uint_as_float(x.x), __uint_as_float(y.x)));
    z.y = __float_as_uint(__fadd_rn(__uint_as_float(x.y), __uint_as_float(y.y)));
    z.z = __float_as_uint(__fadd_rn(__uint_as_float(x.z), __uint_as_float(y.z)));
    z.w = __float_as_uint(__fadd_rn(__uint_as_float(x.w), __uint_as_float(y.w)));
    return z;
}

}  // namespace box

extern "C" __global__ void __launch_bounds__(256) ardelva_box(box::Args a) {
    using namespace box;
    __shared__ uint32_t bad;
    const uint32_t tid = threadIdx.x;
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + tid;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t packs = a.nbytes / 16;
    uint32_t *ctrl = (uint32_t *)a.region;
    const uint4 *in = (const uint4 *)a.in;

    // an exchange that failed on any rank ends every later one: their bytes would be wrong
    if (tid == 0) bad = ld_sys(ctrl);
    __syncthreads();
    if (bad != 0u) return;

    // 1. this block's share of the bytes into this rank's outbox, then the block's flag after them
    uint4 *mine = slot(a, a.rank);
    for (uint64_t i = index; i < packs; i += stride) mine[i] = in[i];
    __syncthreads();
    if (tid == 0) {
        fence_sys();
        st_release_sys(flag(a, a.rank), a.seq);
    }
    // the local part of a gather while the other ranks' writes travel (in place when `in` is that part of `out`)
    if (a.mode == kGather) {
        uint4 *own = (uint4 *)(a.out + (uint64_t)a.rank * a.nbytes);
        if (own != in)
            for (uint64_t i = index; i < packs; i += stride) own[i] = in[i];
    }

    // 2. the same block of every other rank: it wrote exactly the bytes this block reads
    if (tid == 0) {
        const uint64_t start = timer_ns();
        for (uint32_t r = 0; r < a.world && bad == 0u; ++r) {
            if (r == a.rank) continue;
            const uint32_t *theirs = flag(a, r);
            uint32_t spins = 0, why = 0;
            while (ld_acquire_sys(theirs) != a.seq) {
                if ((++spins & 63u) != 0u) continue;
                if (ld_sys(ctrl) != 0u) why = kAborted;
                else if (timer_ns() - start > a.timeout_ns) why = kTimedOut;
                if (why != 0u) break;
            }
            if (why != 0u) {
                bad = 1u;
                uint32_t *rec = ctrl + kCtrlRank + 8u * a.rank;
                st_sys(rec + 1, a.seq);
                st_sys(rec + 2, r);
                st_sys(rec + 3, blockIdx.x);
                st_sys(rec + 4, why);
                st_sys(rec, 1u);
                st_sys(ctrl, 1u);
            }
        }
    }
    __syncthreads();
    if (bad != 0u) return;

    // 3. the other ranks' bytes out of their outboxes
    if (a.mode == kGather) {
        for (uint32_t r = 0; r < a.world; ++r) {
            if (r == a.rank) continue;
            const uint4 *theirs = slot(a, r);
            uint4 *to = (uint4 *)(a.out + (uint64_t)r * a.nbytes);
            for (uint64_t i = index; i < packs; i += stride) to[i] = ld_sys_v4(theirs + i);
        }
    } else {
        // fp32, added in rank order on every rank: the same bits on all of them
        uint4 *out = (uint4 *)a.out;
        for (uint64_t i = index; i < packs; i += stride) {
            uint4 acc = a.rank == 0u ? in[i] : ld_sys_v4(slot(a, 0) + i);
            for (uint32_t r = 1; r < a.world; ++r) acc = add_f32(acc, r == a.rank ? in[i] : ld_sys_v4(slot(a, r) + i));
            out[i] = acc;
        }
    }
}
