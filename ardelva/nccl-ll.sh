#!/usr/bin/env bash
# Builds ardelva/nccl_ll_tuner.c (vLLM's minimal NCCL tuner, unchanged) and prints the two settings that put it in
# use, for `eval`:   eval "$(ardelva/nccl-ll.sh)"   in the shell that starts each rank.
# The tuner makes NCCL pick ring/LL for a single-node all-reduce of at most LL_TUNER_MAX_BYTES (default 327679);
# NCCL_P2P_LEVEL=SYS lets NCCL write GPU to GPU across CPU sockets. An NCCL_P2P_LEVEL already set is kept.
set -euo pipefail
src="$(cd "$(dirname "$0")" && pwd)/nccl_ll_tuner.c"
tuner="${XDG_CACHE_HOME:-$HOME/.cache}/tensorfold/libnccl_ll_tuner.so"
if [[ ! -f "$tuner" || "$src" -nt "$tuner" ]]; then
    mkdir -p "$(dirname "$tuner")"
    gcc -fPIC -shared -O2 -o "$tuner" "$src"
fi
echo "export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}"
echo "export NCCL_TUNER_PLUGIN=$tuner"
