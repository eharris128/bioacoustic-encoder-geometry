#!/bin/bash
# Round B + Tier 2/3 chain runner for the 24-hour campaign.
#
# Phase 1 (parallel):
#   1a. step11 — Round B core (3.4, 3.5, 3.6) — CPU-heavy, ~3-6 hr.
#   1b. seed=7 random-init activation extraction — GPU, ~6 hr.
# Phase 2: step8 on seed=7 shards — ~1 hr (3.7 cheap mitigation).
# Phase 3: Tier 2/3 expansive experiments (run as time allows):
#   3a. step13 — §4.5 mixing-ratio sweep on sl_eat_bio_ssl_all L9.
#   3b. step14 — multi-class Order INLP symmetric test (closes §4.12 caveat 2).
#   3c. seed=13 extraction + step8 on seed=13 (closes 3.7 fully).
#   3d. step15 — layer-resolved Class-first iter sweep (full 13 layers).
#   3e. step16 — probe-class robustness (linear SVM, ridge, MLP at all cells).
#
# Each phase logs to its own file; chain-level events go to round_b_chain.log.

set -u
cd ~/sentient-futures
source venv/bin/activate

CHAIN_LOG="round_b_chain.log"
log() {
  echo "[round_b $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$CHAIN_LOG"
}

run_step() {
  local name="$1"
  local logfile="$2"
  shift 2
  log "starting $name"
  "$@" > "$logfile" 2>&1
  local rc=$?
  log "$name exit=$rc"
  return $rc
}

PER_ORDER_MANIFEST="artifacts/manifests/naturelm_by_order_p100_m200_n200_20260427T222756Z.jsonl"
SOURCE_MANIFEST="artifacts/manifests/naturelm_by_source_100each_20260418T171459Z.jsonl"

log "=== chain start (24-hour campaign) ==="

# ---------- Phase 1 ----------
log "phase 1a: launching step11 (Round B core) in background"
python -W ignore step11_round_b.py > round_b_step11.log 2>&1 &
STEP11_PID=$!
log "step11 PID=$STEP11_PID"

log "phase 1b: launching seed=7 extraction in background"
python -W ignore collect_esp_aves2_activations.py \
  --manifest "$PER_ORDER_MANIFEST" \
  --models random_init_eat_seed07 \
  --device cuda --skip_existing > extract_seed7_perorder.log 2>&1 &
EXTRACT7_PID=$!
log "seed=7 extract PID=$EXTRACT7_PID"

log "waiting on step11 ..."
wait $STEP11_PID
STEP11_EXIT=$?
log "step11 exit=$STEP11_EXIT"

log "waiting on seed=7 extract ..."
wait $EXTRACT7_PID
EXTRACT7_EXIT=$?
log "seed=7 extract exit=$EXTRACT7_EXIT"

# ---------- Phase 2 ----------
if [ $EXTRACT7_EXIT -eq 0 ]; then
  run_step "step8 on seed=7" step8_seed7.log \
    python -W ignore step8_inlp_aggressive.py \
      --models random_init_eat_seed07 \
      --output_dir artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4/inlp_aggressive_seed7
else
  log "phase 2 (step8 seed=7) skipped: extraction failed"
fi

# ---------- Phase 3 ----------
# 3a — §4.5 mixing-ratio sweep
if [ -f step13_mixing_ratio_sweep.py ]; then
  run_step "step13 §4.5 mixing sweep" step13_mixing.log \
    python -W ignore step13_mixing_ratio_sweep.py
else
  log "step13 mixing sweep script not yet built; skipping for now"
fi

# 3b — multi-class Order INLP symmetric test
if [ -f step14_multiclass_order_inlp.py ]; then
  run_step "step14 multi-class Order INLP" step14_mcorder.log \
    python -W ignore step14_multiclass_order_inlp.py
else
  log "step14 multi-class Order INLP script not yet built; skipping"
fi

# 3c — seed=13 (Round C closer for 3.7)
log "phase 3c: launching seed=13 extraction"
run_step "seed=13 extraction" extract_seed13_perorder.log \
  python -W ignore collect_esp_aves2_activations.py \
    --manifest "$PER_ORDER_MANIFEST" \
    --models random_init_eat_seed13 \
    --device cuda --skip_existing
SEED13_EXIT=$?

if [ $SEED13_EXIT -eq 0 ]; then
  run_step "step8 on seed=13" step8_seed13.log \
    python -W ignore step8_inlp_aggressive.py \
      --models random_init_eat_seed13 \
      --output_dir artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4/inlp_aggressive_seed13

  # Also run step11 on seed=7+13 against seed=42 for cross-seed §4.12 sanity.
  run_step "step11 cross-seed (7,13,42) §4.12 reads" round_b_seeds_71342.log \
    python -W ignore step11_round_b.py \
      --models random_init_eat_seed07 random_init_eat_seed13 random_init_eat_seed42 \
      --output_dir artifacts/comparisons/naturelm_by_order_p100_m200_n200_20260427T222756Z/nway_eat_all4/round_b_random_seeds
fi

# 3d — layer-resolved Class-first iter sweep
if [ -f step15_layer_resolved_inlp.py ]; then
  run_step "step15 layer-resolved iter sweep" step15_layers.log \
    python -W ignore step15_layer_resolved_inlp.py
else
  log "step15 not yet built; skipping"
fi

# 3e — probe-class robustness
if [ -f step16_probe_robustness.py ]; then
  run_step "step16 probe robustness" step16_robust.log \
    python -W ignore step16_probe_robustness.py
else
  log "step16 not yet built; skipping"
fi

log "=== chain complete ==="
