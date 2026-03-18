"""OpenAIProvider — DALL-E 3 image generation via the OpenAI API.

Requires:
  pip install openai
  OPENAI_API_KEY=sk-... (environment variable)

Model IDs:
  dall-e-3   (default, 1024x1024 / 1024x1792 / 1792x1024)
  dall-e-2   (256x256, 512x512, 1024x1024)

Notes:
  - DALL-E 3 ignores seed; set seed in the spec for metadata purposes only.
  - DALL-E 3 does not accept negative_prompt; it is stored in metadata but not sent.
  - num_images > 1 is supported for dall-e-2 but not dall-e-3 (we loop instead).
"""

from __future__ import annotations

import base64
import os
from datetime import datetime, timezone
from pathlib import Path

from image_generation.providers.base import BaseProvider, GeneratedImage, ProviderConfigError


class OpenAIProvider(BaseProvider):

    PROVIDER_NAME = "openai"
    DEFAULT_MODEL = "dall-e-3"
    VALID_SIZES_DALLE3 = {"1024x1024", "1024x1792", "1792x1024"}
    VALID_SIZES_DALLE2 = {"256x256", "512x512", "1024x1024"}

    def __init__(self, client) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> "OpenAIProvider":
        try:
            import openai
        except ImportError:
            raise ProviderConfigError(
                "openai package is not installed. Run: pip install openai"
            )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ProviderConfigError(
                "OPENAI_API_KEY environment variable is not set. "
                "Export your API key before running."
            )
        return cls(openai.OpenAI(api_key=api_key))

    def generate(self, spec, output_dir: Path) -> list[GeneratedImage]:
        from image_generation.specs import ImageGenSpec
        assert isinstance(spec, ImageGenSpec)

        model = spec.model or self.DEFAULT_MODEL
        size = self._resolve_size(spec, model)
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        results: list[GeneratedImage] = []

        for i in range(spec.num_images):
            filename = f"image_{i:02d}.png" if spec.num_images > 1 else "image.png"
            out_path = output_dir / filename

            response = self._client.images.generate(
                model=model,
                prompt=spec.prompt,
                size=size,
                response_format="b64_json",
                n=1,
            )

            image_data = response.data[0]
            img_bytes = base64.b64decode(image_data.b64_json)
            out_path.write_bytes(img_bytes)

            revised_prompt = getattr(image_data, "revised_prompt", None)

            results.append(GeneratedImage(
                path=out_path,
                provider=self.PROVIDER_NAME,
                model=model,
                prompt=spec.prompt,
                negative_prompt=spec.negative_prompt,
                width=spec.width,
                height=spec.height,
                seed=spec.seed,  # informational only for DALL-E 3
                timestamp=timestamp,
                extra={
                    "size": size,
                    "revised_prompt": revised_prompt,
                    "negative_prompt_note": "not sent to API (unsupported by DALL-E 3)",
                },
            ))

        return results

    def _resolve_size(self, spec, model: str) -> str:
        size = spec.size  # e.g. "1024x1024"
        valid = self.VALID_SIZES_DALLE3 if "3" in model else self.VALID_SIZES_DALLE2
        if size not in valid:
            default = "1024x1024"
            print(
                f"Warning: size {size!r} not valid for {model}. "
                f"Valid: {sorted(valid)}. Using {default!r}."
            )
            return default
        return size
