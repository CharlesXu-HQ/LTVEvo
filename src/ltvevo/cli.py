"""Command-line entry point for frozen LTV prediction experiments."""

from __future__ import annotations

import argparse
import json
import os

from .data import prepare
from .provider import ApiProvider
from .search import finalize_search, run_search
from .task import TaskSpec


def _provider(args: argparse.Namespace, *, required: bool) -> ApiProvider | None:
    url = args.provider_url or os.getenv("LTVEVO_PROVIDER_URL")
    model = args.model or os.getenv("LTVEVO_MODEL")
    key = os.getenv("LTVEVO_API_KEY")
    if not any((url, model, key)) and not required:
        return None
    if not url or not model or not key:
        raise ValueError("provider URL, model, and LTVEVO_API_KEY are required together")
    return ApiProvider(url=url, model=model, api_key=key, thinking=args.thinking,
                       send_reasoning_effort=not args.omit_reasoning_effort)


def _experiment_args(parser: argparse.ArgumentParser, *, include_task: bool) -> None:
    if include_task:
        parser.add_argument("--task", required=True)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--journal", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--docker-image", default="ltvevo-sandbox")


def _provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider-url", help="OpenAI-compatible base URL or chat/completions URL")
    parser.add_argument("--model", help="Agent model name")
    parser.add_argument("--thinking", choices=("enabled", "disabled", "omit"),
                        default="enabled")
    parser.add_argument("--omit-reasoning-effort", action="store_true",
                        help="Omit reasoning_effort for providers that reject this field")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ltvevo", description="Agent-led, fixed-horizon customer value experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="Build frozen snapshots from full raw data")
    prepare_parser.add_argument("--task", required=True)
    prepare_parser.add_argument("--output", required=True)

    baseline_parser = commands.add_parser("baseline", help="Evaluate historical and PyTorch seeds")
    _experiment_args(baseline_parser, include_task=True)
    baseline_parser.add_argument("--harness", choices=("none", "model-evo"), default=None)

    search_parser = commands.add_parser("search", help="Run or resume Agent model experiments")
    _experiment_args(search_parser, include_task=True)
    search_parser.add_argument("--harness", choices=("none", "model-evo"), default=None)
    search_parser.add_argument("--steps", type=int, required=True,
                               help="Total Agent experiments, including completed steps")
    _provider_args(search_parser)

    finalize_parser = commands.add_parser("finalize", help="Evaluate selected model on sealed test")
    _experiment_args(finalize_parser, include_task=False)
    _provider_args(finalize_parser)

    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(TaskSpec.load(args.task), args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "baseline":
        result = run_search(args.task, args.snapshots, args.journal, None, 0,
                            device=args.device, docker_image=args.docker_image,
                            harness=args.harness)
        print(json.dumps({"journal": args.journal,
                          "baseline": result["baseline"]["metrics"],
                          "zero_reference": result["zero_reference"]["metrics"],
                          "torch_seed": result["torch_seed"]["metrics"],
                          "best_id": result["best_id"]}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "search":
        result = run_search(args.task, args.snapshots, args.journal,
                            _provider(args, required=args.steps > 0), args.steps,
                            device=args.device, docker_image=args.docker_image,
                            harness=args.harness)
        print(json.dumps({"journal": args.journal, "steps": len(result["steps"]),
                          "best_id": result["best_id"], "stopped": result.get("stop")},
                         ensure_ascii=False, indent=2))
        return 0
    result = finalize_search(args.journal, args.snapshots, device=args.device,
                             docker_image=args.docker_image,
                             provider=_provider(args, required=False))
    print(json.dumps(result["final"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
