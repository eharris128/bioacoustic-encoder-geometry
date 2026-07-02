# Species-Pair Expansion — Phylogenetic Gradient Paper

**Goal:** take the gradient from n=11 → n=19 pairs, chosen to (a) resolve the 2–12.5 MYA
"resolution floor" and (b) fill the empty 15–50 MYA mid-range. All MYA values below are
best-estimates from avian phylogeny and **must be confirmed on TimeTree.org** before use.

## Existing anchors (from context)
| Pair | MYA | Role |
|---|---|---|
| Common Chiffchaff vs Iberian Chiffchaff | ~2 | Floor low end (~chance) |
| House Sparrow vs Tree Sparrow | ~4.4 | Low |
| Bullfinch vs Hawfinch | ~12.5 | Floor high end (mid-layer peak → T11 convergence) |
| Bullfinch vs Tawny Owl | ~60–70 | Cross-order anchor |
| + ~7 others (incl. Great Tit / bokharensis) | — | span TBD |

## 8 new pairs (ordered by ascending MYA)

| # | Species A | Species B | Est. MYA | Divergence level | Why this pair |
|---|---|---|---|---|---|
| 1 | Willow Warbler (*Phylloscopus trochilus*) | Common Chiffchaff (*P. collybita*) | ~6 | Within-genus | Just above the 2 MYA no-separation point — tests where separation first emerges |
| 2 | Common Chaffinch (*Fringilla coelebs*) | Brambling (*F. montifringilla*) | ~7 | Within-genus | Second floor probe; different family than #1 to control for lineage |
| 3 | Common Blackbird (*Turdus merula*) | Song Thrush (*T. philomelos*) | ~8 | Within-genus | Densifies the floor mid-point |
| 4 | Eurasian Blackcap (*Sylvia atricapilla*) | Garden Warbler (*S. borin*) | ~10 | Within-genus | Directly below the Bullfinch/Hawfinch 12.5 point |
| 5 | Great Tit (*Parus major*) | Blue Tit (*Cyanistes caeruleus*) | ~14 | Cross-genus (Paridae) | First point above the floor; starts the mid-range |
| 6 | European Goldfinch (*Carduelis carduelis*) | Common Chaffinch (*Fringilla coelebs*) | ~22 | Cross-subfamily (Fringillidae) | Fills empty 15–25 window |
| 7 | Yellowhammer (*Emberiza citrinella*) | Common Chaffinch (*Fringilla coelebs*) | ~30 | Cross-family (Emberizidae/Fringillidae) | Fills 25–35 window |
| 8 | Carrion Crow (*Corvus corone*) | Common Blackbird (*Turdus merula*) | ~40 | Cross-family (Corvidae/Turdidae) | Fills 35–45 window; bridges to the 60–70 cross-order anchor |

## Resulting gradient coverage
Combined MYA points ≈ 2, 4.4, 6, 7, 8, 10, 12.5, 14, 22, 30, 40, 60–70 (+existing others).
The 2–12.5 floor goes from **2 points → 6 points**; the 15–50 mid-range goes from
**0 points → 3 points**.

## Design rationale
- **Floor first (pairs 1–4):** the resolution floor is the most interesting sub-claim and
  currently rests on two points. Four new within-genus pairs spanning 6–10 MYA turn "two dots"
  into a resolved curve and let you state *where* separation emerges, not just *that* it does.
- **Mid-range (pairs 5–8):** the gradient currently jumps from ~12.5 to ~60. Pairs at
  14/22/30/40 make the accuracy-vs-MYA trend continuous instead of two clusters with a gap.
- **n effect:** 11 → 19 pairs drops the Spearman ρ significance threshold from ρ≥0.59 to
  ρ≈0.46, and gives margin so one noisy pair can't sink the result.
- **Supply is deliberate:** every species here is an extremely common, heavily-recorded
  European bird — high xeno-canto / NatureLM coverage means recording count (n₁) won't be
  the bottleneck for any pair.

## Caveats before running
1. **Confirm every MYA on TimeTree.org** — values above are estimates. Exact medians will shift
   the x-positions but not the design logic.
2. **Confirm species exist in `NatureLM-audio-training`** with ≥ ~30 items each — the loader
   filters on the metadata `species` field, so the string must match that dataset's format.
3. Keep the pipeline identical to the existing 11 (mean-pool, PCA-50, LORO) or all 11 must
   be re-run for comparability.
