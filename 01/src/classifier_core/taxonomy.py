from pathlib import Path
import os
import yaml

ROOT = Path(os.environ.get("CLASSIFIER_HOME", Path(__file__).resolve().parents[2]))


def load_config(name):
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


TAXONOMY = load_config("taxonomy.yaml")
CATEGORIES = tuple(TAXONOMY["categories"])
PRIORITY_MARKERS = tuple(k for k, v in TAXONOMY["markers"].items() if v["tier"])
ACTION_MARKERS = tuple(k for k, v in TAXONOMY["markers"].items() if not v["tier"])
NEXT_STEPS = (
    "escalate_human_urgent",
    "human_review",
    "request_more_info",
    "route_legal",
    "route_finance",
    "auto_reply_cancellation",
    "auto_reply_kb",
    "route_tech_qa",
    "route_support_operations",
    "route_support",
    "route_product",
    "route_product_marketing",
)
