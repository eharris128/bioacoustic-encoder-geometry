# Image Generation

Optional artifact-generation layer for converting research outputs into images.
This is not part of the core scientific pipeline — it is a utility for producing
visual summaries or illustrations derived from experiment results.

---

## Directory Layout

```
image_generation/
  __init__.py
  specs.py                 ImageGenSpec dataclass + from_json_file()
  runner.py                run(spec) → GenerationResult; owns filesystem layout
  cli.py                   argparse CLI; entry point is python -m image_generation.cli
  providers/
    __init__.py            registry + get_provider()
    base.py                BaseProvider ABC + GeneratedImage dataclass
    stub.py                StubProvider — matplotlib placeholder, no credentials
    openai_dalle.py        OpenAIProvider — DALL-E 3/2 via openai package
  examples/
    bullfinch_illustration.json   example spec (stub provider, runs immediately)

artifacts/generated_images/   all generated output lands here
  <name>/
    <timestamp>/
      image.png            (or image_00.png, image_01.png, ...)
      spec.json            the spec that produced this run
      metadata.json        full provenance record
```

---

## Quick Start

**No credentials needed (stub provider):**

```bash
# From a prompt
python -m image_generation.cli --prompt "a bullfinch on a branch" --name my-test

# From the built-in example spec
python -m image_generation.cli --example

# From a spec file
python -m image_generation.cli --spec image_generation/examples/bullfinch_illustration.json

# Dry-run (inspect resolved spec without generating)
python -m image_generation.cli --prompt "..." --name test --dry-run
```

**With OpenAI (DALL-E 3):**

```bash
pip install openai
export OPENAI_API_KEY=sk-...

python -m image_generation.cli \
  --prompt "Eurasian Bullfinch, Victorian natural history illustration" \
  --name bullfinch-dalle \
  --provider openai \
  --model dall-e-3 \
  --size 1024x1024
```

---

## Configuring a Provider

### stub (default)
No configuration required. Generates a matplotlib placeholder image.
Useful for testing the pipeline end-to-end without API access.

### openai
1. `pip install openai`
2. Set `OPENAI_API_KEY` in your environment.
3. Pass `--provider openai`. Default model is `dall-e-3`.

Valid sizes for DALL-E 3: `1024x1024`, `1024x1792`, `1792x1024`.
DALL-E 3 ignores `negative_prompt` and `seed` at the API level
(both are recorded in metadata for reproducibility bookkeeping).

### Adding a new provider
1. Create `image_generation/providers/yourprovider.py`
2. Subclass `BaseProvider`, implement `from_env()` and `generate()`
3. Register in `image_generation/providers/__init__.py`:
   ```python
   from image_generation.providers.yourprovider import YourProvider
   REGISTRY["yourprovider"] = YourProvider
   ```

---

## Spec Format

A spec is a JSON file (or dataclass) describing the full generation request.
All fields are preserved in the saved metadata alongside the output images.

```json
{
  "name": "bullfinch-illustration",
  "prompt": "...",
  "negative_prompt": "",
  "source_artifact": "artifacts/runs/bullfinch-layer11-structure/RUN-000007/outputs/result.json",
  "tags": ["bullfinch", "illustration"],
  "provider": "stub",
  "model": "",
  "width": 1024,
  "height": 1024,
  "num_images": 1,
  "seed": 42,
  "metadata": {}
}
```

`source_artifact` is intentionally a plain string path — it records provenance
without creating a hard dependency between the image system and the experiment pipeline.

---

## What Gets Saved

Each run creates a directory under `artifacts/generated_images/<name>/<timestamp>/`:

| File | Contents |
|---|---|
| `image.png` | Generated image (or `image_00.png`, `image_01.png` for multiple) |
| `spec.json` | The full spec, saved before generation starts |
| `metadata.json` | Provider, model, prompt, seed, file paths, timestamp, success/error |

`spec.json` is written before the API call so that the request is recorded even
if generation fails.

---

## What is Out of Scope for v1

- Batch/queue processing of multiple specs
- Integration with the researchctl job system (can be added later as a job wrapper)
- Image post-processing or compositing
- Fine-tuning or LoRA workflow
- Automatic prompt construction from result.json fields
- Any form of persistent image index or search

---

## Design Principles

- **Explicit artifacts over implicit state** — everything is on disk, human-readable
- **Optional layer** — nothing in the core pipeline imports from `image_generation/`
- **Provider abstraction** — swap backends by changing `provider` in the spec
- **Reproducibility** — seed, model, and full prompt are always recorded
- **Fail clearly** — missing credentials raise `ProviderConfigError` with a clear message
