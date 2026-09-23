#!/usr/bin/env bash
# Run nccl_sweep.py on all 8 Sparks in throwaway containers (needs idle GPUs: stop the server first). Usage: nccl_sweep.sh "NCCL_PROTO=Simple NCCL_MAX_NCHANNELS=16"
set -euo pipefail
cd "$(dirname "$0")/.."; source configs/cluster.env; [ -f configs/cluster.local.env ] && source configs/cluster.local.env
read -ra FA <<<"$FABRIC_A"; MASTER=${DIST_INIT_ADDR%:*}
extra=${1:-}; envs=""; for kv in $NCCL_ENV $extra; do envs+=" -e $kv"; done; envs=${envs// -e SGLANG8_ROCE_ALLREDUCE=1/}
for r in 1 2 3 4 5 6 7; do rsync -a runtime/ "$SSH_USER@${FA[$r]}:${WORKDIR#\$HOME/}/runtime/"; done
for r in 7 6 5 4 3 2 1 0; do
  cmd="docker run --rm --name mimo26-nccl-r$r --gpus all --network host --ipc host --cap-add IPC_LOCK --ulimit memlock=-1 --device /dev/infiniband -v $WORKDIR/runtime:/rt $envs $IMAGE torchrun --nnodes 8 --nproc-per-node 1 --node-rank $r --master-addr $MASTER --master-port 29611 /rt/nccl_sweep.py"
  if [ $r = 0 ]; then bash -c "$cmd" > /tmp/nccl-sweep-r0.log 2>&1; grep "^{" /tmp/nccl-sweep-r0.log; else ssh -o BatchMode=yes $SSH_USER@${FA[$r]} "$cmd" > /tmp/nccl-sweep-r$r.log 2>&1 & fi
done; wait
