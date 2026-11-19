// The `direct` exchange of p2p.cu with a process a rank, as TensorFold runs its ranks: can a kernel of one
// process write into device memory another process allocated on another GPU, and what does an exchange cost
// then? A measuring program, not part of the engine and not a check.
//
//   p2p_ipc DEVICES [REPEATS]      DEVICES is a comma list of 2 to 8 device indices
//
// The parent forks a child a device before anything touches CUDA. Each child allocates its inbox (slots and
// flags) on its own device and publishes a CUDA IPC handle to it (cudaIpcGetMemHandle) in memory the processes
// share; each opens the others' handles (cudaIpcOpenMemHandle, with peer access enabled lazily) and from then on
// its kernels write its words straight into the peers' inboxes. No host thread takes part in an exchange: the
// kernels wait on each other's flags, so the processes only agree on when to begin.
//
// Timed in every rank, reported by rank 0: `one` is an exchange launched and waited for, `queued` is 50 launched
// one after another and waited for once, a fiftieth of it. Every rank checks its result against what the ranks'
// words (which each can compute) add up to.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include "p2p_kernel.cuh"

static const uint64_t kMostWords = (1u << 20) / 4;
static const int kSizes = 5;
static const uint64_t kBytes[kSizes] = {5120, 10240, 40960, 163840, 1u << 20};

// What the processes share: the handles, a counter a phase that every rank bumps when it gets there, a word a
// rank sets when it fails, and rank 0's results.
struct Shared {
    cudaIpcMemHandle_t slots[kMost], flags[kMost];
    std::atomic<int> arrived[64];
    std::atomic<int> failed;
    char why[kMost][160];
};

static uint32_t word_of(int rank, uint64_t i) {
    const float v = (float)((int)((i * 2654435761u + rank * 40503u) >> 12 & 0xfff) - 2048) / 64.0f;
    uint32_t w;
    std::memcpy(&w, &v, 4);
    return w;
}

// Every rank reaches `phase` before any goes on; false if a rank failed or 120 seconds pass.
static bool together(Shared *sh, int phase, int world) {
    sh->arrived[phase].fetch_add(1);
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (sh->arrived[phase].load() < world) {
        if (sh->failed.load() || std::chrono::steady_clock::now() > until) return false;
        usleep(50);
    }
    return !sh->failed.load();
}

#define TRYR(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { std::snprintf(sh->why[rank], sizeof sh->why[rank], "%s: %s", #call, cudaGetErrorString(e_)); sh->failed.store(1); return 1; } } while (0)

static int child(Shared *sh, const std::vector<int> &devs, int rank, int repeats) {
    const int world = (int)devs.size();
    const uint64_t slot_words = kMostWords;
    TRYR(cudaSetDevice(devs[rank]));
    uint32_t *mine, *out, *slots, *flags, *state;
    TRYR(cudaMalloc(&mine, kMostWords * 4));
    TRYR(cudaMalloc(&out, kMostWords * 4 * world));
    TRYR(cudaMalloc(&slots, slot_words * 4 * world * 2));
    TRYR(cudaMalloc(&flags, (size_t)kFlagWords * 4 * world));
    TRYR(cudaMalloc(&state, 256));
    TRYR(cudaMemset(flags, 0, (size_t)kFlagWords * 4 * world));
    TRYR(cudaMemset(state, 0, 256));
    {
        std::vector<uint32_t> host(kMostWords);
        for (uint64_t i = 0; i < kMostWords; ++i) host[i] = word_of(rank, i);
        TRYR(cudaMemcpy(mine, host.data(), kMostWords * 4, cudaMemcpyHostToDevice));
    }
    TRYR(cudaIpcGetMemHandle(&sh->slots[rank], slots));
    TRYR(cudaIpcGetMemHandle(&sh->flags[rank], flags));
    if (!together(sh, 0, world)) return 1;

    uint32_t *peer_slots[kMost] = {}, *peer_flags[kMost] = {};
    for (int p = 0; p < world; ++p) {
        if (p == rank) continue;
        TRYR(cudaIpcOpenMemHandle((void **)&peer_slots[p], sh->slots[p], cudaIpcMemLazyEnablePeerAccess));
        TRYR(cudaIpcOpenMemHandle((void **)&peer_flags[p], sh->flags[p], cudaIpcMemLazyEnablePeerAccess));
    }
    if (!together(sh, 1, world)) return 1;
    if (rank == 0) {
        std::printf("| bytes a rank | what | one: median | p10 | p90 | queued: median | p10 | p90 | right on every rank |\n|---:|---|---:|---:|---:|---:|---:|---:|---|\n");
        std::fflush(stdout);
    }

    uint32_t seq = 0;
    int phase = 2;
    for (int size = 0; size < kSizes; ++size) {
        const uint64_t bytes = kBytes[size], words = bytes / 4;
        const unsigned grid = (unsigned)std::min<uint64_t>(16, (words / 4 + 255) / 256);
        for (uint32_t gather = 0; gather < 2; ++gather) {
            auto launch = [&]() {
                Args a = {};
                a.mine = mine;
                a.out = out;
                for (int p = 0; p < world; ++p) {
                    a.peer_slots[p] = peer_slots[p];
                    a.peer_flags[p] = peer_flags[p];
                }
                a.my_slots = slots;
                a.my_flags = flags;
                a.state = state;
                a.words = words;
                a.slot_words = slot_words;
                a.world = world;
                a.rank = rank;
                a.seq = ++seq;
                a.gather = gather;
                k_exchange<<<grid, 256>>>(a);
            };
            // every rank launches the same exchanges in the same order; the kernels meet through the flags
            if (!together(sh, phase++, world)) return 1;
            const int queue = 50;
            std::vector<double> one, queued;
            for (int it = 0; it < 20 + repeats; ++it) {
                const auto start = std::chrono::steady_clock::now();
                launch();
                TRYR(cudaDeviceSynchronize());
                if (it >= 20) one.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
            }
            if (!together(sh, phase++, world)) return 1;
            for (int it = 0; it < 3 + repeats / 10; ++it) {
                const auto start = std::chrono::steady_clock::now();
                for (int q = 0; q < queue; ++q) launch();
                TRYR(cudaDeviceSynchronize());
                if (it >= 3) queued.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / queue);
            }
            uint32_t st[2];
            TRYR(cudaMemcpy(st, state, 8, cudaMemcpyDeviceToHost));
            bool right = st[1] == 0;
            if (!right) std::snprintf(sh->why[rank], sizeof sh->why[rank], "waited two seconds for a flag at %llu bytes", (unsigned long long)bytes);
            std::vector<uint32_t> got(words * (gather ? world : 1));
            TRYR(cudaMemcpy(got.data(), out, got.size() * 4, cudaMemcpyDeviceToHost));
            for (uint64_t i = 0; i < words && right; ++i) {
                if (gather) {
                    for (int src = 0; src < world; ++src) right = right && got[(uint64_t)src * words + i] == word_of(src, i);
                } else {
                    float sum = 0.0f;
                    for (int src = 0; src < world; ++src) {
                        const uint32_t w = word_of(src, i);
                        float v;
                        std::memcpy(&v, &w, 4);
                        sum += v;
                    }
                    uint32_t want;
                    std::memcpy(&want, &sum, 4);
                    right = got[i] == want;
                }
            }
            if (!right) {
                if (!sh->why[rank][0]) std::snprintf(sh->why[rank], sizeof sh->why[rank], "wrong result at %llu bytes, %s", (unsigned long long)bytes, gather ? "gather" : "sum");
                sh->failed.store(1);
                return 1;
            }
            if (!together(sh, phase++, world)) return 1;  // every rank was right
            if (rank == 0) {
                std::sort(one.begin(), one.end());
                std::sort(queued.begin(), queued.end());
                std::printf("| %llu | %s | %.1f | %.1f | %.1f | %.1f | %.1f | %.1f | yes |\n", (unsigned long long)bytes, gather ? "gather" : "sum",
                            one[one.size() / 2], one[one.size() / 10], one[one.size() * 9 / 10],
                            queued[queued.size() / 2], queued[queued.size() / 10], queued[queued.size() * 9 / 10]);
                std::fflush(stdout);
            }
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: p2p_ipc DEVICES [REPEATS]   (DEVICES: 2 to 8 indices, comma separated)\n");
        return 2;
    }
    std::vector<int> devs;
    std::vector<char> list(argv[1], argv[1] + std::strlen(argv[1]) + 1);
    for (char *tok = std::strtok(list.data(), ","); tok; tok = std::strtok(nullptr, ",")) devs.push_back(std::atoi(tok));
    const int world = (int)devs.size();
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 300;
    if (world < 2 || world > kMost || repeats < 10) {
        std::fprintf(stderr, "p2p_ipc: 2 to 8 devices and at least 10 repeats\n");
        return 2;
    }
    Shared *sh = (Shared *)mmap(nullptr, sizeof(Shared), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (sh == MAP_FAILED) {
        std::perror("mmap");
        return 2;
    }
    std::memset((void *)sh, 0, sizeof(Shared));
    std::printf("### A small exchange with a process a rank: devices %s, a kernel writing into inboxes opened through CUDA IPC, %d measured exchanges each\n\n", argv[1], repeats);
    std::printf("Microseconds, as rank 0 timed them. `one`: an exchange launched and waited for. `queued`: 50 launched one after another, waited for once, a fiftieth of it.\n\n");
    std::fflush(stdout);
    std::vector<pid_t> pids;
    for (int rank = 0; rank < world; ++rank) {
        const pid_t pid = fork();
        if (pid < 0) {
            std::perror("fork");
            return 2;
        }
        if (pid == 0) _exit(child(sh, devs, rank, repeats));
        pids.push_back(pid);
    }
    int bad = 0;
    for (int rank = 0; rank < world; ++rank) {
        int status = 0;
        waitpid(pids[rank], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            ++bad;
            std::printf("\nrank %d (device %d) stopped: %s\n", rank, devs[rank], sh->why[rank][0] ? sh->why[rank] : (WIFSIGNALED(status) ? "killed by a signal" : "another rank failed, or a rank did not arrive in 120 seconds"));
        }
    }
    std::printf("\n");
    return bad ? 1 : 0;
}
