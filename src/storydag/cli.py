"""StoryDAG command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from storydag.pipeline import default_output_dir, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="storydag",
        description="StoryDAG: causal-continuity-preserving novel-to-script pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the full novel-to-script pipeline")
    run_parser.add_argument("--novel", required=True, help="Path to the source novel text file")
    run_parser.add_argument("--title", required=True, help="Script title and output folder name")
    run_parser.add_argument(
        "--output",
        help="Output directory (default: outputs/{title})",
    )
    return parser


def cmd_run(args: argparse.Namespace) -> int:
    novel_path = Path(args.novel)
    if not novel_path.is_file():
        print(f"错误：找不到小说文件 {novel_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else default_output_dir(args.title)
    result = run_pipeline(novel_path, args.title, output_dir=output_dir)

    print(f"因果图: {result.output_dir / 'causal_graph.json'}")
    print(f"剧本:   {result.output_dir / 'script.yaml'}")
    print(f"指标:   {result.output_dir / 'metrics.json'}")
    print(f"CCR:    {result.metrics.ccr:.2%} ({result.metrics.satisfied_edge_count}/{result.metrics.total_edges})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return cmd_run(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
