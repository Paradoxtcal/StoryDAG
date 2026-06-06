"""Allow ``python -m storydag run --novel path.txt``."""

from __future__ import annotations

import sys

from storydag.cli import main

if __name__ == "__main__":
    sys.exit(main())
