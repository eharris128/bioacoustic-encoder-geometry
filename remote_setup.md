# Remote A10 setup — taxonomic scale-up

End-to-end walkthrough for spinning up an A10 GPU instance, downloading
the NatureLM-audio-training dataset fresh, sampling a denser per-Order
manifest, extracting activations for all 5 EAT-family models, and
running the analyses. Then pushing artifacts back to the repo.

## 1. Provision the instance

A10 (24 GB GPU, 30 vCPU, 200 GB RAM, 1.4 TiB SSD, $1.29/hr) is the right
fit. The model is small (12-layer transformer, 768-dim) and there's no
need for the H100 / GH200 capacity. Total wall-clock budget: ~1 hour
including dataset download, so cost is ~$1.50–2 per scale-up iteration.

The 1.4 TiB SSD is comfortable headroom for both:
- the full HF parquet cache (~14 GB once we cache more parts than just `part0`)
- shards for 5 models × 800 samples × ~1.6 GB/100 samples ≈ 65 GB

## 2. Bootstrap the environment

```bash
# On the remote A10
git clone <this-repo-url> sentient-futures
cd sentient-futures

python3 -m venv venv
source venv/bin/activate

# CUDA-enabled torch (default index works for A10 / sm_86).
# torchcodec is required by torchaudio>=2.4 for torchaudio.load — without
# it, every extraction fails with ModuleNotFoundError. Pin transformers
# at 4.57.6 because newer versions break EAT custom-model loading.
pip install torch torchaudio torchcodec
pip install "transformers==4.57.6" huggingface_hub safetensors pyarrow \
            matplotlib scikit-learn scipy timm duckdb pandas

# HF downloads have no read timeout by default, which can cause indefinite
# CLOSE-WAIT hangs on throttled connections. Force a 120s timeout.
echo 'export HF_HUB_DOWNLOAD_TIMEOUT=120' >> venv/bin/activate
```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True NVIDIA A10
```

## 3. Sample the new manifest

`sample_naturelm_by_order.py` walks the full HuggingFace parquet
catalogue (not just `part0/`) via DuckDB, filters by `metadata.class`
and `metadata.order`, and writes a stratified manifest. Defaults match
the teammate's bird-Order panel:

```bash
python sample_naturelm_by_order.py \
    --samples_per_order 100 \
    --samples_mammalia 200 \
    --samples_nonbio 200 \
    --max_rows_to_scan 1500000
```

Output: `artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>.jsonl`.

Expected runtime: ~5–10 minutes (DuckDB streams parquet remotely; each
shard scan is ~10 s on a fast network). Bumps `--max_rows_to_scan` if
buckets stay unfilled.

Verify per-Order coverage:
```bash
python -c "
import json
from collections import Counter
manifest = sorted(__import__('glob').glob('artifacts/manifests/naturelm_by_order_p*.jsonl'))[-1]
buckets = Counter()
with open(manifest) as f:
    for line in f:
        r = json.loads(line)
        buckets[r['bucket']] += 1
print(dict(buckets))
"
```

Should show ~100 per target Order, 200 Mammalia, 200 nonbio.

## 4. Extract activations for all 5 models

The extraction script already supports `--device cuda`. Run for each
model:

```bash
MANIFEST=artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>.jsonl

for MODEL in eat_all eat_bio sl_eat_all_ssl_all sl_eat_bio_ssl_all; do
    python collect_esp_aves2_activations.py \
        --manifest $MANIFEST \
        --models $MODEL \
        --device cuda \
        --dtype float16 \
        --output_dir artifacts/roadmap_part1
done

# Random-init baseline
python collect_esp_aves2_activations.py \
    --manifest $MANIFEST \
    --models random_init_eat_seed42 \
    --device cuda \
    --dtype float16 \
    --output_dir artifacts/roadmap_part1
```

Expected runtime per model: **~1 minute on A10** for 800 samples (vs
~30 minutes on local CPU). Total extraction: **~5 minutes wall clock
for all 5 models**.

## 5. Enrich the new manifest

The manifest already has taxonomic fields baked in (the sampler reads
them from parquet metadata). But the analysis scripts expect a file
named `..._taxonomic.jsonl`. Either:

```bash
# Option A: copy
cp artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>.jsonl \
   artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>_taxonomic.jsonl

# Option B: re-run the enrichment pass (idempotent; verifies coverage)
python enrich_manifest_taxonomy.py \
    --manifest artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>.jsonl \
    --output   artifacts/manifests/naturelm_by_order_p100_m200_n200_<TIMESTAMP>_taxonomic.jsonl
```

## 6. Re-run the analyses against the new manifest

Each analysis script takes a manifest path and a roadmap dir. We need
to update the constants at the top of each script — or pass overrides:

```bash
NEW_MANIFEST_ID=naturelm_by_order_p100_m200_n200_<TIMESTAMP>
ROADMAP=artifacts/roadmap_part1/$NEW_MANIFEST_ID

# Step 2-taxonomic: per-Class + per-Order frame-level metrics
python step2_taxonomic_frame_level.py \
    --tax_manifest artifacts/manifests/${NEW_MANIFEST_ID}_taxonomic.jsonl \
    --roadmap_dir $ROADMAP \
    --output_dir  artifacts/comparisons/$NEW_MANIFEST_ID/nway_eat_all4/taxonomic_frame_level

# Step 3b: species barycenters
python step3b_species_barycenters.py \
    --tax_manifest artifacts/manifests/${NEW_MANIFEST_ID}_taxonomic.jsonl \
    --roadmap_dir $ROADMAP \
    --output_dir  artifacts/comparisons/$NEW_MANIFEST_ID/nway_eat_all4/species_barycenters

# Step 3c: Veitch hierarchy — now meaningful with 4 distinct Orders
python step3c_veitch_hierarchy.py \
    --tax_manifest artifacts/manifests/${NEW_MANIFEST_ID}_taxonomic.jsonl \
    --roadmap_dir $ROADMAP \
    --output_dir  artifacts/comparisons/$NEW_MANIFEST_ID/nway_eat_all4/veitch_hierarchy

# Bootstrap CIs
python step5_bootstrap_taxonomic.py \
    --tax_manifest artifacts/manifests/${NEW_MANIFEST_ID}_taxonomic.jsonl \
    --roadmap_dir $ROADMAP \
    --output_dir  artifacts/comparisons/$NEW_MANIFEST_ID/nway_eat_all4/bootstrap_taxonomic_cis
```

**Note on the Veitch test with 4 Orders.** The current
`step3c_veitch_hierarchy.py` only computes Passeriformes vs other-Aves.
For the denser test, it needs to be extended to compute (Aves − Mammalia)
vs each of (Order_i − Aves) for i ∈ {Passer, Charadrii, Pici, Strigi}.
That's a small edit — happy to land it locally before you push to the
remote, or do it in-place once the data is in.

## 7. Commit and push

The shards are gitignored, so only the analysis CSVs and PNGs come
back. Total artifact size: probably ~30 MB.

```bash
git add artifacts/manifests/naturelm_by_order_*.jsonl \
        artifacts/manifests/naturelm_by_order_*_summary.json \
        artifacts/comparisons/naturelm_by_order_*/

# Note: do NOT add artifacts/roadmap_part1/ — it's gitignored and shards
# are 60+ GB.
git commit -m "Per-Order taxonomic scale-up: 4 bird Orders × 100 samples"
git push
```

## 8. Spin down

```bash
# Optional: copy the new shards somewhere persistent if you want to
# re-analyze later. Cheaper than re-extracting on a fresh instance.
# tar cf - artifacts/roadmap_part1/$NEW_MANIFEST_ID | gzip > shards.tar.gz
# aws s3 cp shards.tar.gz s3://my-bucket/  # or your storage of choice

# Then stop the instance from the cloud console.
```

## Cost summary

| step                          | wall clock      | A10 cost |
|-------------------------------|-----------------|---------:|
| Bootstrap env + clone         | 5 min           |    $0.11 |
| Manifest sampling             | 5–10 min        |    $0.16 |
| Extraction (5 models × 800)   | 5 min           |    $0.11 |
| Taxonomic analyses + bootstrap | 10–15 min       |    $0.27 |
| Push + spin down              | 2 min           |    $0.04 |
| **Total**                     | **~30–40 min**  | **~$0.70** |

About a $1 experiment to unlock a 4-Order Veitch test.

## Caveats and gotchas

- **The HF parquet cache is per-instance.** If you spin up a fresh A10
  later, the cache is gone — re-downloads from scratch. Persist
  somewhere if you'll iterate.
- **`step3c_veitch_hierarchy.py` needs extending for 4 Orders** before
  you can take advantage of denser sampling. Currently it does
  Passer vs other-Aves only. Easy fix; flag it before running.
- **Don't `git add artifacts/roadmap_part1/`** — it's gitignored and
  contains the 60+ GB of shards. The local repo's `.gitignore` should
  prevent this, but worth being explicit on the remote.
- **Analysis scripts have hardcoded `MANIFEST_ID` constants at the top.**
  All major scripts accept `--tax_manifest` and `--roadmap_dir`
  overrides; check the argparse defaults if anything looks off.
