# ardelva

Work in this fork that is not upstream TensorFold's: measuring programs first, then the changes they lead to.
Upstream is https://github.com/ashhart/TensorFold; this fork's `main` follows it and is not edited here, and the
fork's own line is the branch `ardelva`.

## probe

Programs that measure one thing on a machine with GPUs and print a table for a person to read. They are not
checks and decide nothing by themselves; a change to a transport or a kernel is argued from what they print.

| file | what it measures |
| --- | --- |
| `probe/p2p.cu`, `probe/p2p.sh` | a small exchange between the GPUs of one host (every rank ends with the sum, or with every rank's bytes) four ways, timed the same way: `cudaMemcpyPeerAsync`, one kernel a rank writing into the peers' device memory, the same kernel through page-locked host memory (the way the bytes travel in upstream's mailbox), and NCCL (`NCCL=DIR`). One process holds the devices. |
| `probe/tp-suite.sh` | runs upstream's `tf-tp-test suite` (every collective bit for bit, then latency, and its decode window) for a list of devices and a backend, a process a rank, and prints its `RESULT` lines. |
