"""runner.py — executes an ImageGenSpec and persists all artifacts + metadata.

The runner is the only place that knows about the filesystem layout for
generated images. It keeps specs, images, and metadata together so that
any generated artifact is self-describing.

Output layout:
  artifacts/generated_images/<name>/<timestamp>/
    image.png           (or image_00.png, image_01.png for num_images > 1)
    spec.json           the full spec that produced this run
    metadata.json       generation results + file paths + provenance
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from image_generation.providers import get_provider, GeneratedImage
from image_generation.specs import ImageGenSpec

ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "generated_images"


@dataclass
class GenerationResult:
    """Full record of a completed generation run."""
    name: str
    output_dir: Path
    spec_path: Path
    metadata_path: Path
    images: list[GeneratedImage]
    timestamp: str
    success: bool
    error: str = ""

    @property
    def image_paths(self) -> list[Path]:
        return [img.path for img in self.images]


def run(
    spec: ImageGenSpec,
    output_root: Path | None = None,
) -> GenerationResult:
    """Execute spec, save all artifacts, return a GenerationResult.

    Args:
        spec:        the fully-constructed ImageGenSpec to execute.
        output_root: override the default artifacts/generated_images/ root.
                     Useful for tests or custom workflows.

    Raises:
        ProviderConfigError: if provider credentials are missing.
        Any exception from the provider's generate() call.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root or ARTIFACTS_ROOT
    output_dir = root / spec.name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the spec first — even if generation fails, we have a record of the request
    spec_path = output_dir / "spec.json"
    spec_path.write_text(spec.to_json(), encoding="utf-8")

    provider = get_provider(spec.provider)

    images: list[GeneratedImage] = []
    error = ""
    success = False
    try:
        images = provider.generate(spec, output_dir)
        success = True
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        # Always write metadata, even on failure (partial record is useful)
        metadata_path = output_dir / "metadata.json"
        _write_metadata(
            path=metadata_path,
            spec=spec,
            images=images,
            timestamp=timestamp,
            success=success,
            error=error,
        )

    return GenerationResult(
        name=spec.name,
        output_dir=output_dir,
        spec_path=spec_path,
        metadata_path=metadata_path,
        images=images,
        timestamp=timestamp,
        success=success,
        error=error,
    )


def _write_metadata(
    path: Path,
    spec: ImageGenSpec,
    images: list[GeneratedImage],
    timestamp: str,
    success: bool,
    error: str,
) -> None:
    record: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": timestamp,
        "success": success,
        "error": error or None,
        "spec": spec.to_dict(),
        "images": [
            {
                "path": str(img.path),
                "filename": img.path.name,
                "provider": img.provider,
                "model": img.model,
                "prompt": img.prompt,
                "negative_prompt": img.negative_prompt,
                "width": img.width,
                "height": img.height,
                "seed": img.seed,
                "timestamp": img.timestamp,
                **img.extra,
            }
            for img in images
        ],
    }
    path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
