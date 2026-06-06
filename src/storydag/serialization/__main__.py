"""CLI: python -m storydag.serialization script.yaml graph.json [-o metrics.json]"""

from __future__ import annotations

import argparse
import sys

from storydag.serialization.metrics import compute_metrics_from_files, write_metrics_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute StoryDAG causal closure metrics")
    parser.add_argument("script", help="Path to script YAML")
    parser.add_argument("graph", help="Path to causal graph JSON")
    parser.add_argument("-o", "--output", help="Optional path to write metrics JSON")
    args = parser.parse_args(argv)

    report = compute_metrics_from_files(args.script, args.graph)
    if args.output:
        write_metrics_report(report, args.output)
    print(report.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
