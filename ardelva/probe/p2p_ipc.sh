#!/bin/sh
# Build ardelva/probe/p2p_ipc.cu with nvcc and run it for each list of devices given: a process a device, the
# peers' inboxes opened through CUDA IPC, a kernel writing into them. Needs the GPUs.
#   CUDA=/path/to/cuda ardelva/probe/p2p_ipc.sh 0,1 0,4 0,1,4,5
set -eu
here=$(cd "$(dirname "$0")" && pwd)
: "${CUDA:?set CUDA to the toolkit's directory}"
[ $# -ge 1 ] || { echo "usage: p2p_ipc.sh DEVICES [DEVICES ...]   e.g. 0,1 0,4 0,1,4,5" >&2; exit 2; }
out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
"$CUDA/bin/nvcc" -O3 -std=c++17 -arch=sm_${SM:-120} -o "$out/p2p_ipc" "$here/p2p_ipc.cu"
for devices in "$@"; do
    CUDA_DEVICE_ORDER=PCI_BUS_ID timeout "${TIMEOUT:-600}" "$out/p2p_ipc" "$devices" "${REPEATS:-300}" || echo "p2p_ipc $devices: exit $?"
done
