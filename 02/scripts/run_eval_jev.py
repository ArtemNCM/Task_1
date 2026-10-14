import asyncio
from dataclasses import asdict
import os
from pathlib import Path
import sys
import httpx
import yaml
from dotenv import load_dotenv
from classifier_core.taxonomy import ROOT
from classifier_core.eval.runner import arguments, main, hash_object
from classifier_core.eval.dataset import gold_decision
from classifier_jev.questions import build_questions
from classifier_jev.jev_client import JevClient, mock_transport
from classifier_jev.jev_classifier import JevClassifier

JEV_ROOT = Path(__file__).resolve().parents[1]
config = yaml.safe_load((JEV_ROOT / "config/jev.yaml").read_text(encoding="utf-8"))


async def factory(stack, args, cases, settings, prices, cache):
    questions = build_questions(config["questions_version"])
    mock = args.classifier == "mock-jev"
    if args.with_fallback:
        raise ValueError(
            "Jev has shared retries/safety fallback; no cross-paradigm model fallback configured"
        )
    if not mock and args.classifier != "jev-1.13.0":
        raise ValueError("Use jev-1.13.0 or mock-jev")
    load_dotenv(JEV_ROOT / ".env")
    load_dotenv(ROOT / ".env")
    key = "fixture" if mock else os.environ.get(config["key_env"])
    if not key:
        raise ValueError("Configured Jev API key missing")
    if not mock and not config["slug"]:
        raise ValueError("Verify and configure the pinned Jev model slug")
    transport = (
        mock_transport({c.ticket: gold_decision(c) for c in cases}, questions)
        if mock
        else None
    )
    client = await stack.enter_async_context(
        httpx.AsyncClient(
            transport=transport, timeout=settings["failure"]["timeout_seconds"]
        )
    )
    model = {
        "id": args.classifier,
        "slug": config["slug"],
        "params": {"thresholds": config},
        "price_key": args.classifier,
        "provider": "mock" if mock else config["provider"],
    }
    provider = JevClient(client, config, key, questions)
    return JevClassifier(
        [model],
        {args.classifier: provider},
        settings,
        prices,
        prompt_version=config["questions_version"],
        prompt_hash=hash_object(
            {"questions": [asdict(q) for q in questions], "config": config}
        ),
        cache=cache,
        jev_config=config,
    )


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--classifier" not in argv:
        argv = ["--classifier", "mock-jev", *argv]
    try:
        asyncio.run(
            main(arguments(argv), factory, JEV_ROOT / "results", extra_config=config)
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
