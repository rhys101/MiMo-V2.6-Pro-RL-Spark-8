#!/usr/bin/env bash
# Cluster control for MiMo-V2.6-Pro on 8 Sparks. Run on the head node (rank 0) from $WORKDIR.
#   build | distribute | serve | stop | status | logs [rank] | env
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source configs/cluster.env
[ -f configs/cluster.local.env ] && source configs/cluster.local.env
read -ra N <<<"$NODES"; read -ra FA <<<"$FABRIC_A"; read -ra FB <<<"$FABRIC_B"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=10)
on() { local r=$1; shift; if [ "$r" = 0 ]; then bash -c "$*"; else "${SSH[@]}" "$SSH_USER@${FA[$r]}" "$*"; fi; }
rel() { local p=${1#\$HOME/}; echo "${p#$HOME/}"; }   # rsync destination relative to the remote home

fabric_guard() { # refuse bulk transfer unless the route uses a fabric NIC
  local dev; dev=$(ip -o route get "$1" | grep -o 'dev [^ ]*' | cut -d' ' -f2)
  case " $FABRIC_IFACES " in *" $dev "*) ;; *) echo "non-fabric route to $1 via $dev" >&2; exit 9;; esac
}

cmd_build() {
  local b; for i in "${!N[@]}"; do [ "${N[$i]}" = "$BUILD_NODE" ] && b=$i; done
  fabric_guard "${FA[$b]}"
  rsync -a --delete docker patches runtime third_party "$SSH_USER@${FA[$b]}:$(rel "$WORKDIR")/build/"
  "${SSH[@]}" "$SSH_USER@${FA[$b]}" "cd $WORKDIR/build && docker build -f docker/Dockerfile -t $IMAGE . 2>&1 | tail -25"
}

cmd_distribute() { # docker save on build node, stream to every other node over the fabric
  local b; for i in "${!N[@]}"; do [ "${N[$i]}" = "$BUILD_NODE" ] && b=$i; done
  local id; id=$(on "$b" "docker image inspect -f '{{.Id}}' $IMAGE")
  for i in "${!N[@]}"; do
    [ "$i" = "$b" ] && continue
    if [ "$(on "$i" "docker image inspect -f '{{.Id}}' $IMAGE 2>/dev/null" || true)" = "$id" ]; then echo "${N[$i]} has $IMAGE"; continue; fi
    local ip; if (( i % 2 )); then ip=${FB[$i]}; else ip=${FA[$i]}; fi
    ( "${SSH[@]}" "$SSH_USER@${FA[$b]}" "FABRIC_IFACES='$FABRIC_IFACES'; $(declare -f fabric_guard); fabric_guard $ip; docker save $IMAGE | ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com $SSH_USER@$ip docker load" \
      && echo "${N[$i]} loaded" ) &
  done; wait
  for i in "${!N[@]}"; do [ "$(on "$i" "docker image inspect -f '{{.Id}}' $IMAGE")" = "$id" ] || { echo "${N[$i]} image mismatch"; exit 1; }; done
  echo "all nodes: $IMAGE $id"
}

run_rank() {
  local r=$1 envs="" kv
  for kv in $NCCL_ENV $SERVE_ENV ${EXTRA_ENV:-}; do envs+=" -e $kv"; done
  on "$r" "mkdir -p $WORKDIR/cache $WORKDIR/logs && docker rm -f ${CTN_PREFIX}$r >/dev/null 2>&1; docker run -d --name ${CTN_PREFIX}$r \
    --gpus all --network host --ipc host --shm-size 32g --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit nofile=1048576 \
    --device /dev/infiniband -v $MODEL_DIR:/models/MiMo-V2.6-Pro-RL:ro -v $WORKDIR/cache:/root/.cache \
    -e NODE_RANK=$r -e DIST_INIT_ADDR=$DIST_INIT_ADDR $envs $IMAGE bash /opt/mimo26/runtime/serve.sh >/dev/null && echo started ${N[$r]} rank $r"
}

cmd_serve() {
  for i in "${!N[@]}"; do
    if on "$i" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q ."; then echo "${N[$i]}: GPU busy, refusing" >&2; exit 1; fi
  done
  for (( r=${#N[@]}-1; r>=0; r-- )); do run_rank "$r"; done
  echo "waiting for http://127.0.0.1:${API_PORT:-30000}/health ..."
  for _ in $(seq 1 360); do
    curl -sf -m 5 http://127.0.0.1:30000/health >/dev/null && { echo ready; return 0; }
    for i in "${!N[@]}"; do on "$i" "docker inspect -f '{{.State.Running}}' ${CTN_PREFIX}$i" | grep -q true || { echo "rank $i exited" >&2; cmd_logs "$i" | tail -40; exit 1; }; done
    sleep 10
  done; echo "timeout" >&2; exit 1
}

cmd_stop() {
  for i in "${!N[@]}"; do on "$i" "docker rm -f ${CTN_PREFIX}$i >/dev/null 2>&1 || true" & done; wait
  for i in "${!N[@]}"; do for _ in $(seq 1 60); do on "$i" "nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q ." || break; sleep 2; done; done
  echo stopped
}

cmd_status() {
  for i in "${!N[@]}"; do echo "${N[$i]}: $(on "$i" "docker ps -a --filter name=^${CTN_PREFIX}$i\$ --format '{{.Status}}'; free -g | awk '/Mem/{print \"avail \"\$7\"G\"}'" | tr '\n' ' ')"; done
  curl -s -m 5 http://127.0.0.1:30000/v1/models || true; echo
}

cmd_logs() { on "${1:-0}" "docker logs --tail ${TAIL:-200} ${CTN_PREFIX}${1:-0} 2>&1"; }

"cmd_${1:-status}" "${@:2}"
