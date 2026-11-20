# Flash Next's small collectives as one kernel a rank

The ranks' gathers and fp32 sums of at most `TF_FLASHNEXT_COLLECTIVE_BOX_KB` (default 256) KiB a rank go through one
kernel a rank instead of an NCCL call. On by default; `TF_FLASHNEXT_COLLECTIVE_BOX=0` on any rank keeps every
collective on NCCL, as upstream. `TF_FLASHNEXT_COLLECTIVE_SPIN_US` (default 200, 0 for upstream's sleep) is how long
a wait for an NCCL call polls before it sleeps.

## Why

Measured on xzss at `--tp 2` (asterayx/TensorFold#5), a decode forward without drafts:

| | dense whole | dense split, wait spinning |
|---|---:|---:|
| GPU kernel time a forward | 12.56 ms | 10.31 ms |
| collectives a forward | 48 | 192 |
| wall time a forward | 13.7 ms | 14.5 ms |

The split halves what a card reads of the dense projections, and the forward still got slower. The wall time less
the kernel time grew by about 3.1 ms for 144 more collectives, about 21 us each, and the trace's host range of an
NCCL call had a median of about 20 us. NCCL's own kernel for one took 4 to 9 us. So the cost looked like the host's
NCCL call, not the transfer. Both numbers come from runs under a profiler and from a subtraction, not from a direct
measurement of the call.

## What it is

- `zig/kernels/cuda/ardelva_box.cu`: every rank maps one region of pinned host memory; a rank writes its bytes into
  its own outbox, flags them, waits for the other ranks' flags and reads their outboxes. The orderings are
  upstream's `fabric/tp/mailbox.cu`'s. The exchange's number comes from the host.
- `zig/src/families/flashnext/cuda_box.zig`: the region's setup over the rank link (rank 0 names a file under
  /dev/shm, every rank maps it and answers, rank 0 decides for all), and the launch.
- `cuda_collective.zig`: `gather`, `gatherF32` and `sum` go this way when the bytes a rank fit and are whole
  16-byte packs; the reduce-scatter, the broadcast and everything larger stay NCCL's. Which way is by the size
  alone, so every rank chooses alike.

A sum is added in rank order where NCCL adds in its ring's order: the same bits at two ranks, and at more ranks
wherever each value is another than +0 on one rank alone. The engine's one sum across ranks (a decode row's expert
slots) is of that kind.

A failed wait (30 s, or an abort from any rank) leaves a record in the shared page, ends every later exchange on
every rank, and is reported when the pass finishes; the pass's bytes are not used.

Not with `TF_FLASHNEXT_DECODE_GRAPH=1`: a captured launch would replay one exchange's number.

## What is and is not known

Measured on xzss, cards 0,1 at `--tp 2` and 0,1,4,5 at `--tp 4`, short prompts (asterayx/TensorFold#6): token
fingerprints equal to the NCCL path's for all 232 compared requests; with the dense split a decode forward went from
14.5 to 11.8 ms; with the dense layers whole, at two and at four ranks, no measurable change.

Not yet done: a rank killed on purpose mid-request on a GPU, and the full benchmark's long prompts and concurrency.
