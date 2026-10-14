"""Open the local human-review queue without making model requests."""

import runpy
from bootstrap import ROOT

if __name__ == "__main__":
    try:
        runpy.run_path(str(ROOT / "01/scripts/review.py"), run_name="__main__")
    except (EOFError, KeyboardInterrupt):
        print("\nReview cancelled; no correction saved.")
