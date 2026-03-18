"""CLI for image_generation.

Usage examples:
  python -m image_generation.cli --prompt "a bullfinch on a branch" --name test
  python -m image_generation.cli --spec examples/bullfinch_illustration.json
  python -m image_generation.cli --prompt "..." --provider openai --model dall-e-3
  python -m image_generation.cli --spec spec.json --output-dir /tmp/test-images
  python -m image_generation.cli --example          # run the built-in example spec
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from image_generation.providers import REGISTRY
from image_generation.runner import run, ARTIFACTS_ROOT
from image_generation.specs import ImageGenSpec, example_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m image_generation.cli",
        description="Generate images from a prompt or spec file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", metavar="TEXT", help="Direct prompt string.")
    source.add_argument("--spec", metavar="PATH", help="Path to a spec JSON file.")
    source.add_argument(
        "--example",
        action="store_true",
        help="Run the built-in example spec (stub provider, no credentials needed).",
    )

    parser.add_argument("--name", metavar="NAME", default="unnamed",
                        help="Run name used in output directory. Default: unnamed.")
    parser.add_argument(
        "--provider",
        metavar="PROVIDER",
        default="stub",
        choices=sorted(REGISTRY),
        help=f"Image provider. Available: {', '.join(sorted(REGISTRY))}. Default: stub.",
    )
    parser.add_argument("--model", metavar="MODEL", default="",
                        help="Provider-specific model ID. Default: provider default.")
    parser.add_argument("--width", type=int, default=1024, help="Image width in px.")
    parser.add_argument("--height", type=int, default=1024, help="Image height in px.")
    parser.add_argument("--num-images", type=int, default=1, metavar="N",
                        help="Number of images to generate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (for reproducibility).")
    parser.add_argument("--negative-prompt", metavar="TEXT", default="",
                        help="Negative prompt (passed through when provider supports it).")
    parser.add_argument("--source-artifact", metavar="PATH", default="",
                        help="Path to the research artifact this image derives from.")
    parser.add_argument("--tags", metavar="TAG", nargs="*", default=[],
                        help="Space-separated tags.")
    parser.add_argument("--output-dir", metavar="DIR", default=None,
                        help=f"Override output root. Default: {ARTIFACTS_ROOT}")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the resolved spec and exit without generating.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Re-read REGISTRY here so OpenAI shows up if installed after initial import
    from image_generation.providers import REGISTRY as _R
    parser._option_string_actions["--provider"].choices = sorted(_R)

    # --- Build the spec ---
    if args.example:
        spec = example_spec()
        # Allow CLI overrides on top of example
        if args.provider != "stub":
            spec.provider = args.provider
        if args.model:
            spec.model = args.model
        if args.output_dir:
            pass  # handled below
    elif args.spec:
        spec = ImageGenSpec.from_json_file(args.spec)
        # CLI flags override spec fields when explicitly provided
        if args.provider != "stub":
            spec.provider = args.provider
        if args.model:
            spec.model = args.model
    elif args.prompt:
        spec = ImageGenSpec(
            name=args.name,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            source_artifact=args.source_artifact,
            tags=args.tags or [],
            provider=args.provider,
            model=args.model,
            width=args.width,
            height=args.height,
            num_images=args.num_images,
            seed=args.seed,
        )
    else:
        parser.print_help()
        return 1

    output_root = Path(args.output_dir) if args.output_dir else None

    if args.dry_run:
        print("Resolved spec:")
        print(spec.to_json())
        if output_root:
            print(f"Output root: {output_root}")
        return 0

    print(f"Provider:  {spec.provider}")
    print(f"Model:     {spec.model or '(provider default)'}")
    print(f"Name:      {spec.name}")
    print(f"Prompt:    {spec.prompt[:80]}{'...' if len(spec.prompt) > 80 else ''}")
    print(f"Size:      {spec.size}")
    print(f"Images:    {spec.num_images}")
    print(f"Seed:      {spec.seed}")
    print()

    try:
        result = run(spec, output_root=output_root)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Output dir:  {result.output_dir}")
    print(f"Metadata:    {result.metadata_path}")
    for img in result.images:
        print(f"Image:       {img.path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
