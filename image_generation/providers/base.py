"""BaseProvider — abstract interface all image providers must implement."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GeneratedImage:
    """A single generated image and the record of how it was produced."""
    path: Path                       # absolute path to the saved image file
    provider: str
    model: str
    prompt: str
    negative_prompt: str
    width: int
    height: int
    seed: int | None
    timestamp: str                   # ISO-8601 UTC
    extra: dict[str, Any] = field(default_factory=dict)  # provider-specific fields


class BaseProvider(abc.ABC):
    """Minimal interface for image generation backends.

    Implementors must define:
      - from_env()  — classmethod that reads credentials from the environment
                      and raises ProviderConfigError if they are missing
      - generate()  — accepts an ImageGenSpec, returns one GeneratedImage per
                      requested image, files already saved to output_dir
    """

    @classmethod
    @abc.abstractmethod
    def from_env(cls) -> "BaseProvider":
        """Construct from environment variables.

        Raises ProviderConfigError if required credentials are absent.
        """

    @abc.abstractmethod
    def generate(
        self,
        spec: "ImageGenSpec",  # noqa: F821 — forward ref, avoids circular import
        output_dir: Path,
    ) -> list[GeneratedImage]:
        """Generate images according to spec; save them into output_dir.

        Returns one GeneratedImage per image produced.
        """


class ProviderConfigError(RuntimeError):
    """Raised when a provider is not properly configured (e.g. missing API key)."""
