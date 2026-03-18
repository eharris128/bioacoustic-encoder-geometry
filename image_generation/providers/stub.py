"""StubProvider — local test provider, no credentials required.

Generates a simple matplotlib figure containing the prompt text.
Useful for testing the pipeline end-to-end without any API calls.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from image_generation.providers.base import BaseProvider, GeneratedImage


class StubProvider(BaseProvider):
    """Generates placeholder images locally using matplotlib.

    No credentials required. The output is a labelled rectangle with
    the prompt text — useful for exercising the full pipeline without
    burning API credits or requiring network access.
    """

    PROVIDER_NAME = "stub"
    DEFAULT_MODEL = "stub-v1"

    @classmethod
    def from_env(cls) -> "StubProvider":
        return cls()

    def generate(self, spec, output_dir: Path) -> list[GeneratedImage]:
        from image_generation.specs import ImageGenSpec
        assert isinstance(spec, ImageGenSpec)

        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(spec.seed)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        results: list[GeneratedImage] = []

        for i in range(spec.num_images):
            seed_i = int(rng.integers(0, 2**31)) if spec.seed is None else spec.seed + i
            filename = f"image_{i:02d}.png" if spec.num_images > 1 else "image.png"
            out_path = output_dir / filename

            _render_stub_image(
                path=out_path,
                prompt=spec.prompt,
                negative_prompt=spec.negative_prompt,
                name=spec.name,
                seed=seed_i,
                width=spec.width,
                height=spec.height,
                index=i,
            )

            results.append(GeneratedImage(
                path=out_path,
                provider=self.PROVIDER_NAME,
                model=self.DEFAULT_MODEL,
                prompt=spec.prompt,
                negative_prompt=spec.negative_prompt,
                width=spec.width,
                height=spec.height,
                seed=seed_i,
                timestamp=timestamp,
                extra={"stub": True},
            ))

        return results


def _render_stub_image(
    path: Path,
    prompt: str,
    negative_prompt: str,
    name: str,
    seed: int,
    width: int,
    height: int,
    index: int,
) -> None:
    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    rng = np.random.default_rng(seed)
    bg_color = rng.uniform(0.08, 0.18, 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Border
    border = mpatches.FancyBboxPatch(
        (0.02, 0.02), 0.96, 0.96,
        boxstyle="round,pad=0.01",
        linewidth=2,
        edgecolor="white",
        facecolor="none",
    )
    ax.add_patch(border)

    # Header
    ax.text(
        0.5, 0.93, f"[STUB IMAGE] {name}",
        ha="center", va="top", fontsize=11, color="white",
        fontweight="bold", transform=ax.transAxes,
    )

    # Prompt (word-wrapped)
    wrapped = _wrap(prompt, 60)
    ax.text(
        0.5, 0.78, f'"{wrapped}"',
        ha="center", va="top", fontsize=9, color="#dddddd",
        style="italic", transform=ax.transAxes,
        multialignment="center",
    )

    if negative_prompt:
        ax.text(
            0.5, 0.38, f"neg: {_wrap(negative_prompt, 55)}",
            ha="center", va="top", fontsize=7.5, color="#aaaaaa",
            transform=ax.transAxes, multialignment="center",
        )

    # Metadata strip
    ax.text(
        0.5, 0.10,
        f"provider=stub  seed={seed}  size={width}x{height}  index={index}",
        ha="center", va="bottom", fontsize=7.5, color="#888888",
        transform=ax.transAxes,
    )

    plt.tight_layout(pad=0)
    fig.savefig(str(path), dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def _wrap(text: str, width: int) -> str:
    words = text.split()
    lines: list[str] = []
    line: list[str] = []
    n = 0
    for word in words:
        if n + len(word) + (1 if line else 0) > width:
            lines.append(" ".join(line))
            line = [word]
            n = len(word)
        else:
            line.append(word)
            n += len(word) + (1 if len(line) > 1 else 0)
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)
