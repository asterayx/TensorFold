#!/bin/sh
# Build ardelva/probe/p2p.cu with nvcc and run it for each list of devices given: a table a list of what a small
# exchange between those GPUs costs by peer copy, by a kernel writing into the peers' device memory, and by a
# kernel writing into host memory. Needs the GPUs; one process holds them all.
#   CUDA=/path/to/cuda ardelva/probe/p2p.sh 0,1 0,4 0,1,2,3
set -eu
here=$(cd "$(dirname "$0")" && pwd)
: "${CUDA:?set CUDA to the toolkit's directory}"
[ $# -ge 1 ] || { echo "usage: p2p.sh DEVICES [DEVICES ...]   e.g. 0,1 0,4 0,1,2,3" >&2; exit 2; }
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_${SM:-120} -o "$out/p2p" "$here/p2p.cu"
for devices in "$@"; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID "$out/p2p" "$devices" "${REPEATS:-300}" || echo "p2p $devices: exit $?"
done
