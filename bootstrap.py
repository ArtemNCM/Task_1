"""Resolve repository resources without depending on editable installs elsewhere."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
os.environ["CLASSIFIER_HOME"] = str(ROOT / "01")
for folder in ("01/src", "02/src"):
    sys.path.insert(0, str(ROOT / folder))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
