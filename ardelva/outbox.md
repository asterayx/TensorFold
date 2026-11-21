# The outbox backend

`TF_COMM_BACKEND=outbox` is upstream's mailbox for 2 to 8 ranks of one host. Upstream's mailbox takes two.

- A rank writes its shard once, into its own outbox in pinned host memory every GPU maps, and reads the other
  ranks' outboxes (`zig/src/fabric/tp/outbox.cuh`, `outbox.zig`). The kernel `tp_outbox` is built into the same
  image as `tp_mailbox`.
- All-gathers of at most `TF_TP_MAILBOX_MAX_KB` a rank go through it. The default there is 16 KiB on this backend,
  not the mailbox's 256.
- Sums go through it at two ranks only, where they are the mailbox's bits. Past two ranks a sum stays on NCCL, as
  on upstream's multirank RoCE backend: the GPU suite holds a multirank sum to NCCL's bytes.
- Everything else (larger shards, exchanges, send and recv, the barrier) is NCCL's. The backend sets
  `NCCL_P2P_LEVEL=SYS` before NCCL loads unless the environment sets it.

## What is and is not known

Measured on xzss (four RTX PRO 5000, cards 0,1,4,5: two a CPU) with the probe of asterayx/TensorFold#1, which
is not this backend: one write into host memory was the fastest form at 5 and 10 KiB a rank on four cards (7.8 and
8.5 us against NCCL's 15.9), NCCL was the fastest from 40 KiB up, and `NCCL_P2P_LEVEL=SYS` took a 1 MiB all-reduce
from 90 to 64 us.

Not yet run on a GPU: this backend itself. Here it was built (nvcc 13, sm_120) and the host tests pass
(`zig build test-tp`). The check that counts is `tf-tp-test suite` with `TF_COMM_BACKEND=outbox` at two and at
four ranks, which compares every collective byte for byte and prints its latencies.

Flash Next does not use this layer yet: it calls NCCL directly.
