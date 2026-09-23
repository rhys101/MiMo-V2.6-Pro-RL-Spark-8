#!/usr/bin/env bash
# Tree fan-out of the MiMo checkpoint from the head node (rank 0) to the other
# seven nodes over the two RoCE fabric subnets. Every hop checks that its route
# uses a fabric interface and refuses to send over anything else.
#
#   round 1: head -> r1 (subnet A), head -> r2 (subnet B)
#   round 2: head -> r3 (A), head -> r4 (B), r1 -> r5 (A), r1 -> r6 (B), r2 -> r7 (A)
#
# Needs passwordless SSH from the head to every node and from r1/r2 to r5..r7
# on the fabric addresses. Usage: scripts/copy-model.sh   (STREAMS=4 per hop)
set -Eeuo pipefail
cd "$(dirname "$0")/.."
source configs/cluster.env
[ -f configs/cluster.local.env ] && source configs/cluster.local.env
read -ra FA <<<"$FABRIC_A"; read -ra FB <<<"$FABRIC_B"
M=$(eval echo "$MODEL_DIR")
LOG=$(eval echo "$WORKDIR")/logs; mkdir -p "$LOG"
STREAMS=${STREAMS:-4}
SSHOPT="ssh -T -c aes128-gcm@openssh.com -o Compression=no -o BatchMode=yes -o StrictHostKeyChecking=accept-new"

push() { # src_rank dst_ip label
  local src=$1 dst=$2 name=$3
  local guard="dev=\$(ip -o route get $dst | grep -o 'dev [^ ]*' | cut -d' ' -f2); case ' $FABRIC_IFACES ' in *\" \$dev \"*) ;; *) echo NON-FABRIC ROUTE \$dev; exit 9;; esac; "
  local cmd="${guard}mkdir -p $M; cd $M && ls -A | grep -vx .cache | xargs -P $STREAMS -I{} rsync -a --partial -e '$SSHOPT' {} $SSH_USER@$dst:$M/"
  echo "$(date -u +%T) start $name (rank $src -> $dst)"
  if [ "$src" = 0 ]; then bash -c "$cmd"; else ssh -o BatchMode=yes "$SSH_USER@${FA[$src]}" "$cmd"; fi
  echo "$(date -u +%T) done  $name"
}

for r in 1 2 3 4 5 6 7; do ssh -o BatchMode=yes "$SSH_USER@${FA[$r]}" "mkdir -p $M"; done
push 0 "${FA[1]}" rank1 > "$LOG/copy-rank1.log" 2>&1 &
push 0 "${FB[2]}" rank2 > "$LOG/copy-rank2.log" 2>&1 &
wait
push 0 "${FA[3]}" rank3 > "$LOG/copy-rank3.log" 2>&1 &
push 0 "${FB[4]}" rank4 > "$LOG/copy-rank4.log" 2>&1 &
push 1 "${FA[5]}" rank5 > "$LOG/copy-rank5.log" 2>&1 &
push 1 "${FB[6]}" rank6 > "$LOG/copy-rank6.log" 2>&1 &
push 2 "${FA[7]}" rank7 > "$LOG/copy-rank7.log" 2>&1 &
wait
echo "$(date -u +%T) ALL DONE — verify with sha256sum on every node before serving"
