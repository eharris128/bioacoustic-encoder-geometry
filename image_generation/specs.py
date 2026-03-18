"""ImageGenSpec — typed spec for a single image-generation request.

Specs can be constructed in code or loaded from a JSON file.
They are the sole source of truth for what was requested; they travel
alongside the saved artifacts as part of the metadata record.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class ImageGenSpec:
    # --- Identity ---
    name: str                              # human-readable run name, used in output paths
    prompt: str                            # the main generation prompt

    # --- Optional prompt config ---
    negative_prompt: str = ""             # what to avoid (supported by some providers)
    source_artifact: str = ""             # path/id of the research artifact this derives from
    tags: list[str] = field(default_factory=list)

    # --- Provider config ---
    provider: str = "stub"                # "stub" | "openai" | future providers
    model: str = ""                       # provider-specific model id; "" = provider default

    # --- Generation parameters ---
    width: int = 1024
    height: int = 1024
    num_images: int = 1
    seed: int | None = None               # None = provider picks; set for reproducibility

    # --- Freeform metadata passed through to the artifact record ---
    metadata: dict[str, Any] = field(default_factory=dict)

    # --- Convenience property ---
    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ImageGenSpec":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ImageGenSpec":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


def example_spec() -> ImageGenSpec:
    """Return a minimal example spec useful for smoke-testing."""
    return ImageGenSpec(
        name="example-bullfinch-cluster",
        prompt=(
            "A detailed scientific illustration of a Eurasian Bullfinch (Pyrrhula pyrrhula) "
            "perched on a branch, in the style of a Victorian natural history plate, "
            "precise feather detail, muted colours"
        ),
        negative_prompt="blurry, cartoon, photorealistic, low quality",
        source_artifact="artifacts/runs/bullfinch-layer11-structure/RUN-000007/outputs/result.json",
        tags=["bullfinch", "illustration", "layer10-structure"],
        provider="stub",
        model="",
        width=1024,
        height=1024,
        num_images=1,
        seed=42,
        metadata={"experiment": "bullfinch-layer11-structure", "run_id": "RUN-000007"},
    )
