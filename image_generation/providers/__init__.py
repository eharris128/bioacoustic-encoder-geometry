"""Provider registry for image_generation.

Import and register providers here. The runner resolves provider names
through REGISTRY; add new providers by inserting into REGISTRY.
"""

from __future__ import annotations

from image_generation.providers.base import BaseProvider, GeneratedImage
from image_generation.providers.stub import StubProvider

REGISTRY: dict[str, type[BaseProvider]] = {
    "stub": StubProvider,
}

# Register OpenAI provider only if the package is importable
try:
    from image_generation.providers.openai_dalle import OpenAIProvider
    REGISTRY["openai"] = OpenAIProvider
except ImportError:
    pass


def get_provider(name: str) -> BaseProvider:
    """Instantiate a provider by name, loading credentials from the environment."""
    if name not in REGISTRY:
        available = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown provider {name!r}. Available: {available}")
    return REGISTRY[name].from_env()


__all__ = ["BaseProvider", "GeneratedImage", "get_provider", "REGISTRY"]
