#!/bin/bash
# Chain-runner for step8 → step9 → step10, launched after step6_v2 completes.
# Designed to run detached on the remote (sentient) for ~8-12 hours.
# Each step writes its own log file; chain-level events go to chain_log.log.

set -u
cd ~/sentient-futures
source venv/bin/activate

CHAIN_LOG="chain_log.log"

log() {
  echo "[chain $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$CHAIN_LOG"
}

log "chain runner starting"

# Wait for step6_v2 to finish (it's already running when this script starts).
log "waiting for step6_v2 to complete"
while pgrep -f step6_inlp_class_order >/dev/null 2>&1; do
  sleep 30
done
log "step6_v2 done"

log "starting step8 (aggressive multi-class INLP, max_iters=80)"
python -W ignore step8_inlp_aggressive.py > step8_aggressive.log 2>&1
log "step8 exit=$?"

log "starting step9 (INLP-Order-first asymmetric test)"
python -W ignore step9_inlp_order_first.py > step9_order_first.log 2>&1
log "step9 exit=$?"

log "starting step10 (Veitch permutation null, n_perm=200)"
python -W ignore step10_veitch_permutation_null.py > step10_perm_null.log 2>&1
log "step10 exit=$?"

log "chain runner complete"
