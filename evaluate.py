"""Run one model or the fixed five-model comparison; mock is the safe default."""

import argparse
import os
import subprocess
import sys
from bootstrap import ROOT

MODELS = {
    "haiku-5-5": "v6",
    "gpt-6-luna": "v7",
    "gemini-3-8-flash": "v6",
    "sonnet-5-5": "v6",
    "jev-1.13.0": "q4",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=["mock-oracle", "all", *MODELS], default="mock-oracle"
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if args.model != "mock-oracle" and not args.allow_live:
        parser.error("Live evaluations require --allow-live")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "01/src"), str(ROOT / "02/src")])
    for name in MODELS if args.model == "all" else [args.model]:
        jev = name == "jev-1.13.0"
        script = ROOT / (
            "02/scripts/run_eval_jev.py" if jev else "01/scripts/run_eval.py"
        )
        command = [
            sys.executable,
            str(script),
            "--classifier",
            name,
            "--split",
            "all",
            "--i-confirm-frozen",
            "--concurrency",
            "1",
            "--no-cache",
            "--reps",
            str(args.reps),
        ]
        if not jev:
            command += ["--prompt", MODELS.get(name, "v7")]
        if args.allow_live:
            command += ["--allow-live"]
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
