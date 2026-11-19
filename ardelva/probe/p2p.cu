// What one small exchange between GPUs of one host costs, by the way the bytes travel. A measuring program, not
// part of the engine and not a check: it prints a table for a person to read before a transport is written.
//
//   p2p DEVICES [REPEATS]      DEVICES is a comma list of 2 to 8 device indices, e.g. 0,1 or 0,4 or 0,1,2,3
//
// One process holds every device. An exchange is what tensor parallelism does many times a token: every rank
// has `bytes` of fp32 of its own and every rank ends with the sum of all the ranks' in rank order (or, for the
// gather rows, with every rank's bytes). Timed from before the first launch to after every device is waited for.
//
//   peer copy   cudaMemcpyPeerAsync of every rank's bytes to every other rank, then a kernel a rank that adds
//   direct      one kernel a rank: it writes its bytes into an inbox in each other rank's device memory (PCIe
//               peer to peer), sets a flag there, waits for the flags in its own inbox and adds
//   host        the same kernel with the inboxes in page-locked host memory every device maps: the way the
//               bytes travel in upstream's mailbox backend
//   direct, a kernel a peer
//               the writes of `direct` as a kernel a peer, launched one after another, then the kernel that
//               flags, waits and adds. On four cards one kernel writing to three peers was measured about ten
//               times slower at a megabyte than these (asterayx/TensorFold#1); this row asks what that is
//               worth in the whole exchange, launches included
//   host, one outbox
//               a rank writes its bytes once, into host memory every rank reads, in place of once a peer
//   nccl        ncclAllReduce / ncclAllGather on communicators of this process (built with -DWITH_NCCL)
//
// All are timed the same way, twice: `one` is one exchange launched and every device waited for, which
// includes what launching and waiting cost; `queued` is 50 exchanges launched one after another on every
// device and waited for once, divided by 50, which is nearer to what an exchange costs inside a pass.
//
// The kernel's orderings (system-scope loads and stores, the flag after the data, a counter that tells the last
// thread) follow zig/src/fabric/tp/mailbox.cu, which does the host form for two ranks in two processes.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#ifdef WITH_NCCL
// NCCL's few entry points used here, declared so that its header is not needed to build (values as in nccl.h).
extern "C" {
typedef struct ncclComm *ncclComm_t;
int ncclCommInitAll(ncclComm_t *comms, int ndev, const int *devlist);
int ncclCommDestroy(ncclComm_t comm);
int ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count, int datatype, int op, ncclComm_t comm, cudaStream_t stream);
int ncclAllGather(const void *sendbuff, void *recvbuff, size_t sendcount, int datatype, ncclComm_t comm, cudaStream_t stream);
int ncclGroupStart(void);
int ncclGroupEnd(void);
const char *ncclGetErrorString(int result);
}
static const int kNcclFloat32 = 7, kNcclSum = 0;
#endif

#include "p2p_kernel.cuh"

// the adding (or gathering) alone, from slots a peer copy filled: what `peer copy` runs after its copies
__global__ void k_combine(const uint32_t *mine, const uint32_t *slots, uint32_t *out, uint64_t words, uint64_t slot_words,
                          uint32_t world, uint32_t rank, uint32_t gather) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words) return;
    if (gather) {
        for (uint32_t r = 0; r < world; ++r) out[(uint64_t)r * words + i] = r == rank ? mine[i] : slots[(uint64_t)r * slot_words + i];
    } else {
        float sum = 0.0f;
        for (uint32_t r = 0; r < world; ++r) sum += __uint_as_float(r == rank ? mine[i] : slots[(uint64_t)r * slot_words + i]);
        out[i] = __float_as_uint(sum);
    }
}

static void fail(const char *what, cudaError_t err) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    std::exit(2);
}
#define TRY(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) fail(#call, e_); } while (0)

struct Rank {
    int dev;
    uint32_t *mine, *out, *slots, *flags, *state;  // device memory
};

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: p2p DEVICES [REPEATS]   (DEVICES: 2 to 8 indices, comma separated)\n");
        return 2;
    }
    std::vector<int> devs;
    std::vector<char> list(argv[1], argv[1] + std::strlen(argv[1]) + 1);  // strtok cuts the copy, not the argument
    for (char *tok = std::strtok(list.data(), ","); tok; tok = std::strtok(nullptr, ",")) devs.push_back(std::atoi(tok));
    const int world = (int)devs.size();
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 300;
    if (world < 2 || world > kMost || repeats < 10) {
        std::fprintf(stderr, "p2p: 2 to 8 devices and at least 10 repeats\n");
        return 2;
    }
    const uint64_t most_words = (1u << 20) / 4;  // the largest exchange below, a rank
    const uint64_t slot_words = most_words;

    std::printf("### What a small exchange costs: devices %s, %d measured exchanges each\n\n", argv[1], repeats);
    bool direct_ok = true;
    for (int a = 0; a < world; ++a) {
        cudaDeviceProp prop;
        TRY(cudaGetDeviceProperties(&prop, devs[a]));
        std::printf("- rank %d: device %d, %s, PCI %04x:%02x:%02x; peer access to the others:", a, devs[a], prop.name, prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
        TRY(cudaSetDevice(devs[a]));
        for (int b = 0; b < world; ++b) {
            if (a == b) continue;
            int can = 0;
            TRY(cudaDeviceCanAccessPeer(&can, devs[a], devs[b]));
            const cudaError_t err = can ? cudaDeviceEnablePeerAccess(devs[b], 0) : cudaErrorPeerAccessUnsupported;
            const bool ok = err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled;
            std::printf(" %d %s", devs[b], ok ? "yes" : (can ? "refused" : "no"));
            if (!ok) direct_ok = false;
            cudaGetLastError();
        }
        std::printf("\n");
    }
    std::printf("\nMicroseconds. `one`: an exchange launched and every device waited for. `queued`: 50 launched one after another, waited for once, a fiftieth of it.\n\n");

    // device memory of each rank, and host inboxes every device maps
    std::vector<Rank> ranks(world);
    std::vector<uint32_t *> host_slots(world), host_flags(world);
    std::vector<std::vector<uint32_t>> host_mine(world, std::vector<uint32_t>(most_words));
    for (int r = 0; r < world; ++r) {
        Rank &k = ranks[r];
        k.dev = devs[r];
        TRY(cudaSetDevice(k.dev));
        TRY(cudaMalloc(&k.mine, most_words * 4));
        TRY(cudaMalloc(&k.out, most_words * 4 * world));
        TRY(cudaMalloc(&k.slots, slot_words * 4 * world * 2));
        TRY(cudaMalloc(&k.flags, (size_t)kFlagWords * 4 * world));
        TRY(cudaMalloc(&k.state, 256));
        TRY(cudaMemset(k.flags, 0, (size_t)kFlagWords * 4 * world));
        TRY(cudaMemset(k.state, 0, 256));
        for (uint64_t i = 0; i < most_words; ++i) {
            const float v = (float)((int)((i * 2654435761u + r * 40503u) >> 12 & 0xfff) - 2048) / 64.0f;
            std::memcpy(&host_mine[r][i], &v, 4);
        }
        TRY(cudaMemcpy(k.mine, host_mine[r].data(), most_words * 4, cudaMemcpyHostToDevice));
        TRY(cudaHostAlloc(&host_slots[r], slot_words * 4 * world * 2, cudaHostAllocPortable | cudaHostAllocMapped));
        TRY(cudaHostAlloc(&host_flags[r], (size_t)kFlagWords * 4 * world, cudaHostAllocPortable | cudaHostAllocMapped));
        std::memset(host_flags[r], 0, (size_t)kFlagWords * 4 * world);
    }

#ifdef WITH_NCCL
    std::vector<ncclComm_t> comms(world);
    {
        const int rc = ncclCommInitAll(comms.data(), world, devs.data());
        if (rc != 0) {
            std::fprintf(stderr, "ncclCommInitAll: %s\n", ncclGetErrorString(rc));
            return 2;
        }
    }
    const int ways = 6;
#else
    const int ways = 5;
#endif

    std::printf("| bytes a rank | what | way | one: median | p10 | p90 | queued: median | p10 | p90 | right |\n|---:|---|---|---:|---:|---:|---:|---:|---:|---|\n");
    uint32_t seq = 0;
    const uint64_t sizes[] = {5120, 10240, 40960, 163840, 1u << 20};
    for (uint64_t bytes : sizes) {
        const uint64_t words = bytes / 4;
        const unsigned grid = (unsigned)std::min<uint64_t>(16, (words / 4 + 255) / 256);  // a thread a 16-byte pack, 16 blocks at most
        for (uint32_t gather = 0; gather < 2; ++gather) {
            // 0 peer copy; 1 direct: one kernel writes into every peer's device memory; 2 host: the same into
            // every peer's host inbox; 3 direct, a kernel a peer: the writes of 1 as a kernel a peer, one after
            // another; 4 host, one outbox: a rank writes its words once, into host memory every rank reads; 5 nccl
            for (int way = 0; way < ways; ++way) {
                const char *name = way == 0 ? "peer copy" : way == 1 ? "direct" : way == 2 ? "host" : way == 3 ? "direct, a kernel a peer" : way == 4 ? "host, one outbox" : "nccl";
                const bool device_inbox = way == 1 || way == 3;
                if ((way == 0 || device_inbox) && !direct_ok) {
                    std::printf("| %llu | %s | %s | - | - | - | - | - | - | no peer access |\n", (unsigned long long)bytes, gather ? "gather" : "sum", name);
                    continue;
                }
                // one exchange on every device, launched and not waited for
                auto launch = [&]() {
                    ++seq;
                    if (way == 0) {
                        // the copies land in the slots of even parity, which k_combine reads
                        for (int src = 0; src < world; ++src)
                            for (int d = 0; d < world; ++d)
                                if (src != d) {
                                    TRY(cudaSetDevice(devs[d]));
                                    TRY(cudaMemcpyPeerAsync(ranks[d].slots + (uint64_t)src * 2 * slot_words, devs[d], ranks[src].mine, devs[src], bytes, 0));
                                }
                        for (int r = 0; r < world; ++r) {
                            TRY(cudaSetDevice(devs[r]));
                            k_combine<<<(unsigned)((words + 255) / 256), 256>>>(ranks[r].mine, ranks[r].slots, ranks[r].out, words, 2 * slot_words, world, r, gather);
                        }
                    } else if (way == 5) {
#ifdef WITH_NCCL
                        // one thread issues every rank's call: a group, as NCCL asks of a thread with several devices
                        ncclGroupStart();
                        for (int r = 0; r < world; ++r) {
                            TRY(cudaSetDevice(devs[r]));
                            const int rc = gather ? ncclAllGather(ranks[r].mine, ranks[r].out, words, kNcclFloat32, comms[r], 0)
                                                  : ncclAllReduce(ranks[r].mine, ranks[r].out, words, kNcclFloat32, kNcclSum, comms[r], 0);
                            if (rc != 0) {
                                std::fprintf(stderr, "nccl: %s\n", ncclGetErrorString(rc));
                                std::exit(2);
                            }
                        }
                        {
                            const int rc = ncclGroupEnd();
                            if (rc != 0) {
                                std::fprintf(stderr, "ncclGroupEnd: %s\n", ncclGetErrorString(rc));
                                std::exit(2);
                            }
                        }
#endif
                    } else {
                        for (int r = 0; r < world; ++r) {
                            Args a = {};
                            a.mine = ranks[r].mine;
                            a.out = ranks[r].out;
                            for (int p = 0; p < world; ++p) {
                                a.peer_slots[p] = device_inbox ? ranks[p].slots : host_slots[way == 4 ? 0 : p];
                                a.peer_flags[p] = device_inbox ? ranks[p].flags : host_flags[way == 4 ? 0 : p];
                            }
                            a.my_slots = device_inbox ? ranks[r].slots : host_slots[way == 4 ? 0 : r];
                            a.my_flags = device_inbox ? ranks[r].flags : host_flags[way == 4 ? 0 : r];
                            a.shared = way == 4;
                            a.sent = way == 3;
                            a.state = ranks[r].state;
                            a.words = words;
                            a.slot_words = slot_words;
                            a.world = world;
                            a.rank = r;
                            a.seq = seq;
                            a.gather = gather;
                            TRY(cudaSetDevice(devs[r]));
                            if (way == 3)
                                for (int p = 0; p < world; ++p)
                                    if (p != r) k_send<<<grid, 256>>>(a, (uint32_t)p);
                            k_exchange<<<grid, 256>>>(a);
                        }
                    }
                };
                auto wait_all = [&]() {
                    for (int r = 0; r < world; ++r) {
                        TRY(cudaSetDevice(devs[r]));
                        TRY(cudaDeviceSynchronize());
                    }
                };
                auto any_failed = [&]() {
                    bool failed = false;
                    if (way >= 1 && way <= 4)
                        for (int r = 0; r < world; ++r) {
                            uint32_t state[2];
                            TRY(cudaSetDevice(devs[r]));
                            TRY(cudaMemcpy(state, ranks[r].state, 8, cudaMemcpyDeviceToHost));
                            if (state[1]) failed = true;
                        }
                    return failed;
                };
                const int queue = 50;
                std::vector<double> one, queued;
                bool failed = false;
                for (int it = 0; it < 20 + repeats && !failed; ++it) {
                    const auto start = std::chrono::steady_clock::now();
                    launch();
                    wait_all();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    if (it >= 20) one.push_back(t);
                    failed = any_failed();
                }
                for (int it = 0; it < 3 + repeats / 10 && !failed; ++it) {
                    const auto start = std::chrono::steady_clock::now();
                    for (int q = 0; q < queue; ++q) launch();
                    wait_all();
                    const double t = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                    if (it >= 3) queued.push_back(t / queue);
                    failed = any_failed();
                }
                if (failed) {
                    std::printf("| %llu | %s | %s | - | - | - | - | - | - | a rank waited two seconds for a flag |\n", (unsigned long long)bytes, gather ? "gather" : "sum", name);
                    for (int r = 0; r < world; ++r) {
                        TRY(cudaSetDevice(devs[r]));
                        TRY(cudaMemset(ranks[r].state, 0, 256));
                    }
                    continue;
                }
                // every rank's result against the host's
                bool right = true;
                std::vector<uint32_t> got(words * world);
                for (int r = 0; r < world && right; ++r) {
                    TRY(cudaSetDevice(devs[r]));
                    TRY(cudaMemcpy(got.data(), ranks[r].out, words * 4 * (gather ? world : 1), cudaMemcpyDeviceToHost));
                    for (uint64_t i = 0; i < words && right; ++i) {
                        if (gather) {
                            for (int src = 0; src < world; ++src) right = right && got[(uint64_t)src * words + i] == host_mine[src][i];
                        } else {
                            float sum = 0.0f;
                            for (int src = 0; src < world; ++src) {
                                float v;
                                std::memcpy(&v, &host_mine[src][i], 4);
                                sum += v;
                            }
                            uint32_t want;
                            std::memcpy(&want, &sum, 4);
                            // NCCL adds the ranks in an order of its own: its sum is held to a rounding of ours
                            float have;
                            std::memcpy(&have, &got[i], 4);
                            right = way == 5 ? std::fabs(have - sum) <= 1e-4f * (1.0f + std::fabs(sum)) : got[i] == want;
                        }
                    }
                }
                std::sort(one.begin(), one.end());
                std::sort(queued.begin(), queued.end());
                std::printf("| %llu | %s | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | %s |\n", (unsigned long long)bytes, gather ? "gather" : "sum", name,
                            one[one.size() / 2], one[one.size() / 10], one[one.size() * 9 / 10],
                            queued[queued.size() / 2], queued[queued.size() / 10], queued[queued.size() * 9 / 10], right ? "yes" : "NO");
                std::fflush(stdout);
            }
        }
    }
#ifdef WITH_NCCL
    for (ncclComm_t comm : comms) ncclCommDestroy(comm);
#endif
    std::printf("\n");
    return 0;
}
