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

### Decision (2026-04-26): scope is the ESP-AVES2 `eat` family only

The four target models for the roadmap pilot are all ESP-AVES2 (HuBERT-base architecture, 12 layers, 768-dim) on Hugging Face under `EarthSpeciesProject/`:

| Model (HF repo) | Status as of 2026-04-26 |
|---|---|
| `esp-aves2-eat-all` | re-published 2026-04-20, ~370 MB safetensors (was zero-tensor placeholder on 2026-04-18) |
| `esp-aves2-eat-bio` | re-published 2026-04-20, ~370 MB safetensors (was zero-tensor placeholder on 2026-04-18) |
| `esp-aves2-sl-eat-all-ssl-all` | working since 2026-04-18, cached locally |
| `esp-aves2-sl-eat-bio-ssl-all` | working since 2026-04-18, cached locally |

The placeholder issue is tracked by [earthspecies/avex#181](https://github.com/earthspecies/avex/issues/181) / fixed by [PR #183](https://github.com/earthspecies/avex/pull/183). The PR itself was still open as of 2026-04-22, but the corrected safetensors were re-uploaded to HF on 2026-04-20 ahead of the merge — no client-side patch is required.

### Out of scope for this pilot
- **Original AVES** (`aves-base-core/bio/nonbio/all`, the legacy torchaudio `.pt` checkpoints under `storage.googleapis.com/esp-public-files/ported_aves/`). `aves-base-all` and `aves-base-nonbio` happen to be on disk under `models/` from earlier exploratory work, but they are not part of the ESP-AVES2 comparison.
- **BirdAVES** (`biox-base`, `biox-large`, `bioxn-large`).

Revisit only if the ESP-AVES2 `eat` sweep raises a question that requires comparing against the legacy training recipes.

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

### Model coverage status

- `esp-aves2-sl-eat-bio-ssl-all` and `esp-aves2-sl-eat-all-ssl-all` — load and extract since 2026-04-18.
- `esp-aves2-eat-bio` and `esp-aves2-eat-all` — re-uploaded to HF on 2026-04-20 with non-empty safetensors (~370 MB each). Pulling into local cache on 2026-04-26 to verify end-to-end. All four `eat`-family models are in scope; see Section 2 for the explicit decision.
