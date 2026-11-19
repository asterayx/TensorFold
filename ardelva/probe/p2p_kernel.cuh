// The exchange kernel the probes share (p2p.cu: one process holds every device; p2p_ipc.cu: a process a device).
// Its orderings follow zig/src/fabric/tp/mailbox.cu.
#pragma once
#include <cstdint>

#include <cuda_runtime.h>

static const int kMost = 8;
static const int kFlagWords = 32;  // a flag to a 128-byte line

struct Args {
    const uint32_t *mine;          // this rank's words (device)
    uint32_t *out;                 // the sum, or world x words gathered
    uint32_t *peer_slots[kMost];   // rank r's inbox data as this device reaches it: [sender][seq & 1][words]
    uint32_t *peer_flags[kMost];   // rank r's inbox flags: [sender * kFlagWords]
    const uint32_t *my_slots;      // this rank's inbox data
    const uint32_t *my_flags;      // this rank's inbox flags
    uint32_t *state;               // device: [0] threads done writing, [1] failed
    uint64_t words;
    uint64_t slot_words;
    uint32_t world, rank, seq, gather;
    uint32_t sent;                 // k_send has already put the words: k_exchange only flags, waits and reads
    uint32_t shared;               // every rank's slots and flags are one buffer: the words are written once
    uint32_t wt;                   // the words go with __stwt (st.global.wt; on sm_120 STG.E.128.STRONG.SYS), not a plain store
    uint32_t unroll;               // a thread loads four packs, then stores four, in place of one and one
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
__device__ __forceinline__ uint32_t add_gpu(uint32_t *p, uint32_t v) {
    uint32_t old;
    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(v) : "memory");
    return old;
}
__device__ __forceinline__ uint64_t timer_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ uint4 ld_sys_v4(const uint32_t *p) {
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void fence_sys() { asm volatile("fence.sc.sys;" ::: "memory"); }

// This rank's words into rank r's inbox: two slots a sender, by the exchange's parity, so that the next
// exchange's words do not land on words still read.
__device__ __forceinline__ void put(const Args &a, uint32_t r) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint4 *slot = (uint4 *)(a.peer_slots[r] + ((uint64_t)a.rank * 2 + (a.seq & 1)) * a.slot_words);
    const uint4 *mine = (const uint4 *)a.mine;
    const uint64_t packs = a.words / 4;
    if (!a.unroll) {
        for (uint64_t i = index; i < packs; i += stride) {
            const uint4 v = mine[i];
            if (a.wt) __stwt(slot + i, v);
            else slot[i] = v;
        }
        return;
    }
    // four loads, then four stores: more writes on their way at once (the packs past a multiple of four go one
    // a thread, after)
    const uint64_t whole = packs & ~(uint64_t)3;
    for (uint64_t i = index * 4; i < whole; i += stride * 4) {
        const uint4 v0 = mine[i], v1 = mine[i + 1], v2 = mine[i + 2], v3 = mine[i + 3];
        if (a.wt) {
            __stwt(slot + i, v0);
            __stwt(slot + i + 1, v1);
            __stwt(slot + i + 2, v2);
            __stwt(slot + i + 3, v3);
        } else {
            slot[i] = v0;
            slot[i + 1] = v1;
            slot[i + 2] = v2;
            slot[i + 3] = v3;
        }
    }
    if (index < packs - whole) {
        const uint4 v = mine[whole + index];
        if (a.wt) __stwt(slot + whole + index, v);
        else slot[whole + index] = v;
    }
}

// The words to one peer and nothing else. A kernel that writes to several peers at once was measured many
// times slower on four cards than a kernel a peer one after another (the same bytes), so a rank may launch
// this once a peer and then k_exchange with `sent` set.
__global__ void k_send(Args a, uint32_t r) { put(a, r); }

// As mailbox.cu does it: the words go in 16-byte stores, a block's first thread counts the block done, the
// last block's fences and sets the flags, and one thread a block waits for the peers' flags while the rest of
// the block waits for it. `words` is a multiple of 4.
__global__ void k_exchange(Args a) {
    const uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t packs = a.words / 4;
    if (!a.sent)
        for (uint32_t r = 0; r < a.world; ++r) {
            if (r == a.rank) continue;
            put(a, r);
            if (a.shared) break;  // one buffer: any peer's pointer is it
        }
    __syncthreads();
    __shared__ uint32_t failed;
    if (threadIdx.x == 0) {
        failed = 0;
        // the block that is last to finish tells every rank the words are there
        if (add_gpu(a.state, 1) + 1 == gridDim.x) {
            fence_sys();
            for (uint32_t r = 0; r < a.world; ++r) {
                if (r == a.rank) continue;
                st_release_sys(a.peer_flags[r] + (uint64_t)a.rank * kFlagWords, a.seq);
                if (a.shared) break;
            }
            st_sys(a.state, 0);
        }
        // wait for every other rank's flag, at most two seconds; at least this exchange's number, since the
        // sender may already have flagged its next
        const uint64_t until = timer_ns() + 2000000000ull;
        for (uint32_t r = 0; r < a.world && !failed; ++r) {
            if (r == a.rank) continue;
            uint32_t polls = 0;
            while (ld_acquire_sys(a.my_flags + (uint64_t)r * kFlagWords) < a.seq) {
                if ((++polls & 63u) == 0u && timer_ns() > until) {
                    failed = 1;
                    a.state[1] = 1;
                    break;
                }
            }
        }
    }
    __syncthreads();
    if (failed) return;
    for (uint64_t i = index; i < packs; i += stride) {
        const uint4 own = ((const uint4 *)a.mine)[i];
        if (a.gather) {
            for (uint32_t r = 0; r < a.world; ++r)
                ((uint4 *)(a.out + (uint64_t)r * a.words))[i] =
                    r == a.rank ? own : ld_sys_v4(a.my_slots + ((uint64_t)r * 2 + (a.seq & 1)) * a.slot_words + i * 4);
        } else {
            float x = 0.0f, y = 0.0f, z = 0.0f, w = 0.0f;
            for (uint32_t r = 0; r < a.world; ++r) {
                const uint4 v = r == a.rank ? own : ld_sys_v4(a.my_slots + ((uint64_t)r * 2 + (a.seq & 1)) * a.slot_words + i * 4);
                x += __uint_as_float(v.x);
                y += __uint_as_float(v.y);
                z += __uint_as_float(v.z);
                w += __uint_as_float(v.w);
            }
            ((uint4 *)a.out)[i] = make_uint4(__float_as_uint(x), __float_as_uint(y), __float_as_uint(z), __float_as_uint(w));
        }
    }
}
