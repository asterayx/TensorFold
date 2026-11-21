#!/bin/sh
# Run TensorFold's own transport runner (tf-tp-test suite: every collective bit for bit, then latency) for one
# list of devices and one backend, a process a rank, and print its RESULT, SUMMARY and FAIL lines.
#   ardelva/suite.sh BIN OUTDIR BACKEND DEVICES [PORT]
#   BIN      zig-out/bin/tf-tp-test (zig build tf-tp-test -Doptimize=ReleaseSafe -Dtp-nvcc=$CUDA/bin/nvcc)
#   BACKEND  nccl | mailbox | outbox   (mailbox is two ranks only; outbox 2 to 8)
#   DEVICES  comma list of device indices, a rank each, e.g. 0,1 or 0,4 or 0,1,2,3
set -u
bin=${1:?bin}; out=${2:?outdir}; backend=${3:?backend}; devices=${4:?devices}; port=${5:-29600}
mkdir -p "$out"
world=$(echo "$devices" | tr ',' '\n' | wc -l)
name="suite-$backend-$(echo "$devices" | tr ',' '_')"
rank=0
for device in $(echo "$devices" | tr ',' ' '); do
    ( CUDA_DEVICE_ORDER=PCI_BUS_ID TF_TP_WORLD=$world TF_TP_RANK=$rank TF_TP_DEVICE=$device TF_TP_PORT=$port TF_COMM_BACKEND=$backend \
        NCCL_DEBUG=${NCCL_DEBUG:-WARN} timeout "${TIMEOUT:-900}" "$bin" suite; echo "EXIT rank $rank rc $?" ) > "$out/$name.r$rank.log" 2>&1 &
    rank=$((rank + 1))
done
wait
echo "== $name"
grep -h "^INFO\|^EXIT\|^SUMMARY\|^FAIL\|^REFUSED" "$out/$name".r*.log | cut -c1-400
grep -h "^RESULT" "$out/$name.r0.log"
