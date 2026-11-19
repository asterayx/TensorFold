# Flash Next at --tp 3 to 8 on CUDA

Upstream runs Flash Next on one GPU or at `--tp 2`. This line lets `--tp` go to 8: a process a rank, started as at
`--tp 2` (`--tp N --rank r --master HOST --master-port PORT`, rank 0 first or last, it waits).

## What the ranks share

Past two ranks the dense split does not apply (two KV heads do not divide by four), so every rank holds the whole
dense layers and computes them itself, as with `TF_FLASHNEXT_SPLIT_DENSE=0` at two ranks. What is spread is what
upstream already spreads at `--tp 2`:

- the routed experts: 512 / N whole experts a rank. A decode row's slots are summed over the ranks in fp32; a
  slot is another value than +0 on one rank alone, so the order of the sum cannot change a bit.
- a prompt pass wider than a decode round: each rank sums the slots of its block of rows (rows / N, rounded up)
  after a reduce-scatter, then the sums are gathered.
- the shared expert and the MTP head: rank 0 alone, as before.

The control link is a star: rank 0 accepts every other rank, each names its rank in its hello, and rank 0's
commands go to all of them and are acknowledged by all.

## What does not change with it

`TF_FLASHNEXT_SHARD_HEAD=1` and the dense split stay two-rank features; at other worlds the first is refused and
the second is off.

## What is and is not known

Verified without a GPU: `zig build test-flashnext-ranks-host` (the link to three followers, the session frames, the
exchange's rows and calls at 2, 3, 4 and 8 ranks), the module's 287 host tests, and the CUDA server compiles.

Not yet run on a GPU: everything else. The checks that count are that `--tp 4` gives the token streams `--tp 2`
gives for the same requests, and what it does to speed. Nothing here says it is faster: every rank still computes
every dense layer, and only the experts' work is divided further.
