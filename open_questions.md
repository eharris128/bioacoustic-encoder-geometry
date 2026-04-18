# Open Questions

## Reference Materials

- Roadmap PDF: [references/roadmaps/aves2_interp_roadmap.pdf](references/roadmaps/aves2_interp_roadmap.pdf)
- Raphael result screenshot 1: [references/raphael_results/raphael_result_2026-04-11_142344.png](references/raphael_results/raphael_result_2026-04-11_142344.png)
- Raphael result screenshot 2: [references/raphael_results/raphael_result_2026-04-11_142359.png](references/raphael_results/raphael_result_2026-04-11_142359.png)
- Raphael result screenshot 3: [references/raphael_results/raphael_result_2026-04-11_142425.png](references/raphael_results/raphael_result_2026-04-11_142425.png)
- Raphael result screenshot 4: [references/raphael_results/raphael_result_2026-04-11_142434.png](references/raphael_results/raphael_result_2026-04-11_142434.png)

## 1. What samples do we want (or not want) from NatureLM-audio-training?

The dataset has 26.4M samples across multiple source datasets and task types. We need to decide what to pull for the activation statistics work.

### Source datasets available
- **Xeno-canto** — bird recordings with full taxonomy (class, order, family, genus, species)
- **iNaturalist** — broader biodiversity (mammals, amphibians, insects)
- **Watkins** — marine mammals
- **WavCaps** — general audio with captions (tattoo guns, traffic, etc. — no taxonomy)
- **NatureLM** — includes NSynth instrument samples (keyboard, etc.)
- **Animal Sound Archive** — unknown coverage

### Decisions needed

**Do we want non-animal audio?**
- WavCaps and NatureLM/NSynth include human-made sounds (instruments, machines, speech)
- Useful as contrast/out-of-distribution baseline (similar to our piano/violin experiments)
- But may dilute the analysis if we're focused on animal vocalization representations
- Roadmap says "compare nature sounds to other sound" — suggests yes, at least some

**Which taxonomic levels to target?**
- The roadmap asks for probes at Class (Aves/Mammalia), Order (Passeriformes/Strigiformes/etc.), and Species levels
- Need sufficient samples per group — how many per category is enough?
- Some orders/species will be heavily overrepresented (Passeriformes dominates Xeno-canto)

**How many samples total?**
- 100-200 per group is likely enough for activation statistics (L2 norms, SVD, intrinsic dim)
- Probes (Section 2 of roadmap) may need more — TBD
- Budget: ~1 hr on CPU for 1000 samples, ~20 min on A10

**Duration filtering?**
- Dataset ranges from 0.2s to 2950s
- Very short clips (<1s) give very few AVES frames (~50)
- Very long clips (>60s) are slow to process and may contain mixed content
- Suggest filtering to 5-60s?

**Task filtering?**
- Dataset has 44 task types (taxonomic-classification, species-detection, caption-generation, Q&A, etc.)
- The same audio appears under multiple tasks with different labels
- Should we deduplicate by `file_name` to avoid processing the same audio twice?

**Mean-pool vs frame-level?**
- Roadmap suggests both
- Mean-pooling: one 768-dim vector per sample per layer — simpler, fits in memory easily
- Frame-level: thousands of 768-dim vectors per sample — much richer but huge storage
- Start with mean-pool, add frame-level for specific case studies?

## 2. Which AVES models to compare?

The roadmap mentions "the 4 transformer models." These are the 4 original AVES models, all HuBERT-base (12 layers, 768-dim, ~95M params), differing only in training data:

| Model | Training Data | Hours | Download |
|-------|--------------|-------|----------|
| aves-base-core | FSD50K + AudioSet core | 153 | `storage.googleapis.com/esp-public-files/ported_aves/aves-base-core.torchaudio.pt` |
| aves-base-bio | core + animal-focused AudioSet/VGGSound | 360 | `storage.googleapis.com/esp-public-files/ported_aves/aves-base-bio.torchaudio.pt` |
| aves-base-nonbio | core + non-animal AudioSet/VGGSound | 360 | `storage.googleapis.com/esp-public-files/ported_aves/aves-base-nonbio.torchaudio.pt` |
| aves-base-all | core + all AudioSet/VGGSound | 5,054 | `storage.googleapis.com/esp-public-files/ported_aves/aves-base-all.torchaudio.pt` |

Currently only `aves-base-all` is downloaded locally. Need to download the other 3 (~360MB each).

Config files exist locally for `aves-base-all`. The same config should work for all 4 (identical architecture). Confirm with mentor.

### Also available (not in roadmap scope)
- **BirdAVES-biox-base** — 12 layers, 768-dim, trained on bio + Xeno-canto (2,570 hr)
- **BirdAVES-biox-large** — 24 layers, 1024-dim, 316M params
- **BirdAVES-bioxn-large** — 24 layers, 1024-dim, + iNaturalist data

### Open questions
- Do we include BirdAVES models or stick to the original 4?
- Should we download all models to the A10 instance to save local disk?

## 3. Storage for activations

The roadmap asks: "Where do we store activations?"
- Local disk? (fine for prototyping)
- Shared storage accessible by the team?
- Format: .npy, .pt, or HDF5?

### Current pilot decision

- Frozen sample manifests live in `artifacts/manifests/`
- Raw activation shards live in `artifacts/roadmap_part1/<manifest_id>/<model>/shards/`
- Shards are `.pt` files containing:
  - `activations`: `(N, 13, 513, 768)` tensors
  - `samples`: manifest metadata plus extraction metadata (`row_index`, `source_dataset`, `valid_token_count`, etc.)
- Keep raw activations private and out of git; only commit code, manifests if useful, and summaries

### Current blocker on model coverage

- `EarthSpeciesProject/esp-aves2-sl-eat-bio-ssl-all` and `EarthSpeciesProject/esp-aves2-sl-eat-all-ssl-all` currently load and are usable for extraction
- `EarthSpeciesProject/esp-aves2-eat-bio` and `EarthSpeciesProject/esp-aves2-eat-all` currently expose placeholder safetensors exports with zero tensors on Hugging Face as of 2026-04-18
- So the practical pilot is `2` working models now, with code paths ready for all `4` once the upstream `eat_*` repos are fixed
