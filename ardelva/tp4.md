# Flash Next at --tp 3 to 8 on CUDA

Upstream runs Flash Next on one GPU or at `--tp 2`. This line lets `--tp` go to 8: a process a rank, started as at
`--tp 2` (`--tp N --rank r --master HOST --master-port PORT`, rank 0 first or last, it waits).

## What the ranks share

**At four ranks the dense split applies, as at two** (`TF_FLASHNEXT_SPLIT_DENSE=0` turns it off): a rank holds and
computes a quarter of every layer.

- DeltaNet layers: 4 of the 16 key heads and 12 of the 48 value heads a rank (kernels `fn_gdn_*_h4`, the `_h8`
  twins' templates at 4 and 12).
- Attention layers: 6 of the 24 query heads a rank, and the one KV head they read. The checkpoint has two KV
  heads, each read by 12 queries, so ranks 0 and 1 both hold the first and ranks 2 and 3 the second: that head's
  projection rows and its cache exist twice. (vLLM replicates KV heads the same way when the ranks outnumber them.)
- Out projections and the shared expert: every rank gathers the ranks' head outputs and computes its own quarter
  of the output columns with whole dot products, then the columns are gathered. No partial sums are added across
  ranks, so the bytes do not depend on the rank count.
- The routed experts: 128 whole experts a rank, summed over the ranks in fp32 where each value is another than +0
  on one rank alone.
- The MTP head and the vocabulary projection: rank 0 alone. `TF_FLASHNEXT_SHARD_HEAD=1` is still a two-rank
  feature (248,320 rows do not cut into four blocks of whole 256-row tiles) and is refused at four.

At 3 or 5 to 8 ranks the heads do not divide, the split is off and every rank computes the whole dense layers; only
the experts are spread.

The control link is a star: rank 0 accepts every other rank, each names its rank in its hello, and rank 0's
commands go to all of them and are acknowledged by all.

## What is and is not known

Measured on xzss (asterayx/TensorFold#3, #6), four ranks with the dense layers whole: token fingerprints equal to two
ranks', and no faster than two ranks (14.0 ms a decode forward against 13.7): every card reads every dense weight.
Two ranks with the split, the collectives' kernel path and the vocabulary shard: 11.3 ms.

Not yet run on a GPU: the split at four ranks. Verified without one: the head ranges, the out projection's and
shared expert's launches at four ranks, the `_h4` kernels' build; the module's host tests pass. The checks that
count are token fingerprints equal to two ranks' for the same requests, then the forward's time.
