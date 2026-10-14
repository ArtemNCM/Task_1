import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid
import httpx
from dotenv import load_dotenv
from ..taxonomy import ROOT, TAXONOMY, load_config
from ..providers.mock import MockProvider
from ..providers.openrouter import OpenRouterProvider
from ..classifier import LLMClassifier
from ..prompt_builder import load_prompt
from ..cache import Cache
from ..hitl import append_review
from .dataset import load_dataset, gold_decision
from .split import get_split, digest
from .metrics import sliced_metrics, stability
from .report import render_table


def hash_object(data):
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", default="mock-oracle")
    parser.add_argument(
        "--prompt", default="v2", help="Versioned prompt file, e.g. v2 or v4"
    )
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--measure-latency", action="store_true")
    parser.add_argument("--with-fallback", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--dataset", type=Path, action="append")
    parser.add_argument("--i-confirm-frozen", action="store_true")
    parser.add_argument(
        "--allow-live",
        action="store_true",
        help="Explicitly authorize billable inference",
    )
    parser.add_argument("--safety-net-always-on", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args(argv)
    if args.reps < 1 or args.concurrency < 1:
        parser.error("reps and concurrency must be positive")
    if args.split in ("holdout", "all") and not args.i_confirm_frozen:
        parser.error("holdout/all requires --i-confirm-frozen")
    return args


def select_cases(args):
    original = ROOT / "data/obrio_task1_70_tickets.json"
    if not original.exists():
        raise ValueError("Place obrio_task1_70_tickets.json in 01/data before running")
    base = load_dataset(original)
    split = get_split(ROOT / "data/split.json", base)
    paths = args.dataset or [original]
    all_cases = load_dataset(*paths)
    # Exact original cases retain holdout protection even when copied to a custom dataset.
    original_by_text = {c.ticket: c.id for c in base}
    selected = []
    for case in all_cases:
        original_id = original_by_text.get(case.ticket)
        if original_id is None:
            if args.split != "holdout":
                selected.append(case)
        elif args.split == "all" or original_id in split[args.split]:
            selected.append(case)
    if not selected:
        raise ValueError("No cases selected")
    return selected, paths, split


async def run_evaluation(cases, classifier, output, meta, concurrency=4, queue=None):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "meta.json", meta | {"complete": False})
    semaphore = asyncio.Semaphore(concurrency)
    predictions = {}

    async def one(case):
        async with semaphore:
            result = await classifier.classify(case.ticket)
            # Persist immediately, before the rest of the batch finishes.
            row = {"case": case.model_dump(), "result": result.model_dump()}
            with (output / "predictions.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            predictions[case.id] = result
            if queue:
                append_review(queue, case, result)

    await asyncio.gather(*(one(case) for case in cases))
    results = [predictions[c.id] for c in cases]
    summary = sliced_metrics(cases, results)
    summary["evidence"] = (
        "MOCK: infrastructure test only" if meta["is_mock"] else "Live API run"
    )
    summary["scope"] = (
        "Full set secondary table; not headline"
        if meta["split"] == "all"
        else meta["split"]
    )
    write_json(output / "summary.json", summary)
    (output / "table.md").write_text(
        render_table(cases, results, summary), encoding="utf-8"
    )
    write_json(output / "meta.json", meta | {"complete": True})
    return predictions, summary


async def main(args=None, factory=None, output_root=None, extra_config=None):
    args = args or arguments()
    settings = load_config("settings.yaml")
    cfg = load_config("models.yaml")
    prices = load_config("prices.yaml")
    if args.clear_cache:
        Cache(ROOT / "data/cache.sqlite").clear()
        return
    settings["safety_net_always_on"] = args.safety_net_always_on
    cases, paths, split = select_cases(args)
    prompt, prompt_hash = load_prompt(args.prompt)
    is_mock = args.classifier.startswith("mock-")
    if not is_mock and not args.allow_live:
        raise ValueError("Live inference requires --allow-live; no request sent")
    cache_enabled = not (args.no_cache or args.measure_latency or args.reps > 1)
    cache = Cache(ROOT / "data/cache.sqlite") if cache_enabled else None
    models = []
    if not factory:
        if args.classifier not in cfg["models"]:
            raise ValueError("Unknown classifier")
        names = [args.classifier] + (
            cfg["fallback_chain"] if args.with_fallback else []
        )
        models = [cfg["models"][name] for name in dict.fromkeys(names)]
        if any(m["provider"] != "mock" for m in models):
            if not args.allow_live:
                raise ValueError("Live fallback requires --allow-live")
            if any(
                not m["slug"] or not m["verified"]
                for m in models
                if m["provider"] != "mock"
            ):
                raise ValueError(
                    "Verify model slugs and set verified=true before inference"
                )
    run_group = uuid.uuid4().hex[:12]
    runs = []
    summaries = []
    async with AsyncExitStack() as stack:
        if factory:
            classifier = await factory(stack, args, cases, settings, prices, cache)
            models = classifier.models
            prompt_hash = classifier.prompt_hash
        else:
            providers = {}
            for model in models:
                if model["provider"] == "mock":
                    providers[model["id"]] = MockProvider(
                        {c.ticket: gold_decision(c) for c in cases},
                        noisy=model["id"] == "mock-noisy",
                    )
                else:
                    load_dotenv(ROOT / ".env")
                    key = os.environ.get("OPENROUTER_API_KEY")
                    if not key:
                        raise ValueError("OPENROUTER_API_KEY missing")
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(
                            timeout=settings["failure"]["timeout_seconds"]
                        )
                    )
                    providers[model["id"]] = OpenRouterProvider(
                        client, cfg["base_url"], key
                    )
            classifier = LLMClassifier(
                models,
                providers,
                settings,
                prices,
                args.prompt,
                prompt,
                prompt_hash,
                cache,
            )
        for rep in range(1, args.reps + 1):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            meta = {
                "timestamp": stamp,
                "classifier": args.classifier,
                "is_mock": is_mock,
                "prompt": classifier.prompt_version,
                "prompt_hash": prompt_hash,
                "model_id": models[0]["id"],
                "params": models,
                "config_hash": hash_object(
                    {
                        "models": cfg,
                        "settings": settings,
                        "prices": prices,
                        "extra": extra_config,
                    }
                ),
                "dataset_hash": hash_object([digest(p) for p in paths]),
                "split_hash": digest(ROOT / "data/split.json"),
                "taxonomy_version": TAXONOMY["version"],
                "taxonomy_hash": hash_object(TAXONOMY),
                "split": args.split,
                "rep": rep,
                "run_group": run_group,
                "cache_enabled": cache_enabled,
                "with_fallback": args.with_fallback,
                "settings": settings,
                "prices": prices,
                "questions_version": getattr(classifier, "questions_version", None),
                "extra_config": extra_config,
            }
            if args.split in ("holdout", "all"):
                with (ROOT / "results/holdout_runs.log").open(
                    "a", encoding="utf-8"
                ) as file:
                    file.write(
                        json.dumps(
                            {
                                "timestamp": stamp,
                                "classifier": args.classifier,
                                "prompt_hash": prompt_hash,
                                "split": args.split,
                            }
                        )
                        + "\n"
                    )
            root = output_root or ROOT / "results"
            output = (
                root
                / f"{stamp}_{args.classifier}_{classifier.prompt_version}_{args.split}_rep{rep}"
            )
            pred, summary = await run_evaluation(
                cases,
                classifier,
                output,
                meta,
                args.concurrency,
                ROOT / "review_queue.jsonl",
            )
            runs.append(pred)
            summaries.append(summary)
            print(
                f"{output.name}: {len(pred)} persisted; case_pass={summary['overall']['case_pass']:.4f}"
                + (" [MOCK]" if is_mock else "")
            )
        values = [s["overall"]["case_pass"] for s in summaries]
        write_json(
            (output_root or ROOT / "results") / f"{run_group}_stability.json",
            stability(runs)
            | {
                "accuracy_mean": sum(values) / len(values),
                "accuracy_range": [min(values), max(values)],
                "reps": len(values),
                "is_mock": is_mock,
            },
        )


def cli():
    try:
        asyncio.run(main())
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from None
