#!/usr/bin/env bash
# Master orchestration for the remote A10 GPU instance.
#
# Designed to run detached (nohup) and survive SSH session drops. Each
# stage is idempotent — skips if its output already exists. Logs to
# ~/sentient-futures/orchestrate.log.
#
# Usage on the A10:
#   cd ~/sentient-futures
#   nohup bash orchestrate_remote.sh > orchestrate.log 2>&1 &
#   disown

set -u  # NOT set -e — we want to continue past failures, not abort

cd "$(dirname "$0")"
source venv/bin/activate

MANIFEST_ID="naturelm_by_order_p100_m200_n200_20260427T222756Z"
MANIFEST="artifacts/manifests/${MANIFEST_ID}.jsonl"
TAX_MANIFEST="artifacts/manifests/${MANIFEST_ID}_taxonomic.jsonl"
ROADMAP="artifacts/roadmap_part1/${MANIFEST_ID}"
NWAY="artifacts/comparisons/${MANIFEST_ID}/nway_eat_all4"

mkdir -p "$NWAY"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log_failure() {
  local stage="$1"
  local err="$2"
  printf '\n## %s — %s FAILED\n```\n%s\n```\n' "$(ts)" "$stage" "$err" >> ~/sentient-futures/FAILURES.md
}

wait_for_other_extractions() {
  while pgrep -f 'collect_esp_aves2_activations.py' > /dev/null; do
    if [ "$(pgrep -f 'collect_esp_aves2_activations.py' | tr '\n' ' ')" != "$$  " ]; then
      echo "[$(ts)] waiting for prior extraction process to finish..."
      sleep 30
    else
      break
    fi
  done
}

echo "=== [$(ts)] orchestrate_remote.sh starting ==="
echo "MANIFEST=$MANIFEST"
echo "ROADMAP=$ROADMAP"
echo "NWAY=$NWAY"
echo

# --- Phase 1: extraction for the 5 models -----------------------------
for MODEL in eat_all eat_bio sl_eat_all_ssl_all sl_eat_bio_ssl_all random_init_eat_seed42; do
  shards_dir="$ROADMAP/$MODEL/shards"
  if [ -d "$shards_dir" ] && [ "$(find "$shards_dir" -name 'shard_*.pt' 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$(ts)] $MODEL: already has shards in $shards_dir, skipping"
    continue
  fi

  wait_for_other_extractions

  echo "=== [$(ts)] Phase 1: extracting $MODEL ==="
  err_file="/tmp/orchestrate_${MODEL}.err"
  if python -W ignore collect_esp_aves2_activations.py \
      --manifest "$MANIFEST" \
      --models "$MODEL" \
      --device cuda --dtype float16 \
      --output_dir artifacts/roadmap_part1 2>"$err_file"
  then
    echo "[$(ts)] $MODEL extraction OK"
  else
    echo "[$(ts)] $MODEL extraction FAILED — see $err_file"
    log_failure "Phase 1: $MODEL extraction" "$(tail -40 "$err_file")"
  fi
done

# --- Phase 2: analyses against the new manifest ----------------------
# The new manifest already has taxonomic fields baked in, so just copy.
if [ ! -f "$TAX_MANIFEST" ]; then
  cp "$MANIFEST" "$TAX_MANIFEST"
  echo "[$(ts)] copied $MANIFEST -> $TAX_MANIFEST"
fi

run_phase() {
  local stage="$1"
  local outdir="$2"
  shift 2
  if [ -d "$outdir" ] && [ "$(ls "$outdir" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[$(ts)] $stage: $outdir exists and is non-empty, skipping"
    return 0
  fi
  echo "=== [$(ts)] $stage ==="
  err_file="/tmp/orchestrate_${stage//[ \/]/_}.err"
  if "$@" 2>"$err_file"; then
    echo "[$(ts)] $stage OK"
  else
    echo "[$(ts)] $stage FAILED — see $err_file"
    log_failure "$stage" "$(tail -40 "$err_file")"
  fi
}

run_phase "Phase 2a: per-source frame-level" \
  "$NWAY/per_source_frame_level" \
  python -W ignore step2_per_source_frame_level.py \
    --roadmap_dir "$ROADMAP" \
    --output_dir "$NWAY/per_source_frame_level"

run_phase "Phase 2b: taxonomic frame-level" \
  "$NWAY/taxonomic_frame_level" \
  python -W ignore step2_taxonomic_frame_level.py \
    --tax_manifest "$TAX_MANIFEST" \
    --roadmap_dir "$ROADMAP" \
    --output_dir "$NWAY/taxonomic_frame_level"

run_phase "Phase 2c: species barycenters" \
  "$NWAY/species_barycenters" \
  python -W ignore step3b_species_barycenters.py \
    --tax_manifest "$TAX_MANIFEST" \
    --roadmap_dir "$ROADMAP" \
    --output_dir "$NWAY/species_barycenters"

run_phase "Phase 2d: Veitch hierarchy (Passer vs other-Aves)" \
  "$NWAY/veitch_hierarchy" \
  python -W ignore step3c_veitch_hierarchy.py \
    --tax_manifest "$TAX_MANIFEST" \
    --roadmap_dir "$ROADMAP" \
    --output_dir "$NWAY/veitch_hierarchy"

run_phase "Phase 2e: late-layer collapse" \
  "$NWAY/late_layer_collapse" \
  python -W ignore step5_late_layer_collapse.py \
    --roadmap_dir "$ROADMAP" \
    --nway_dir "$NWAY" \
    --output_dir "$NWAY/late_layer_collapse"

run_phase "Phase 2f: bootstrap CIs taxonomic" \
  "$NWAY/bootstrap_taxonomic_cis" \
  python -W ignore step5_bootstrap_taxonomic.py \
    --tax_manifest "$TAX_MANIFEST" \
    --roadmap_dir "$ROADMAP" \
    --output_dir "$NWAY/bootstrap_taxonomic_cis"

# --- Phase 3+: deferred experiments (require new scripts authored locally) -
# These wait for scripts to land via git pull.
echo "=== [$(ts)] Phase 3: deferred experiments ==="

for SCRIPT_NAME in step3c_veitch_4order step3b_within_class step5_l12_direction; do
  if [ -f "${SCRIPT_NAME}.py" ]; then
    outdir_name=$(echo "$SCRIPT_NAME" | sed 's/^step[0-9a-z_]*_//' | tr -d '_')
    outdir="$NWAY/${outdir_name}"
    run_phase "Phase 3: $SCRIPT_NAME" \
      "$outdir" \
      python -W ignore "${SCRIPT_NAME}.py" \
        --tax_manifest "$TAX_MANIFEST" \
        --roadmap_dir "$ROADMAP" \
        --output_dir "$outdir"
  else
    echo "[$(ts)] Phase 3: $SCRIPT_NAME.py not yet authored, skipping"
  fi
done

echo "=== [$(ts)] orchestrate_remote.sh finished ==="
touch ~/sentient-futures/orchestrate.done
