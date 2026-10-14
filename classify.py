"""Classify one ticket. Live requests require explicit opt-in."""

import argparse
import asyncio
from contextlib import AsyncExitStack
import json
import os
from pathlib import Path

from bootstrap import ROOT
import httpx
import yaml
from classifier_core.classifier import LLMClassifier
from classifier_core.eval.runner import hash_object
from classifier_core.hitl import append_review
from classifier_core.prompt_builder import load_prompt
from classifier_core.providers.openrouter import OpenRouterProvider
from classifier_core.schema import Case, Expected
from classifier_core.taxonomy import load_config
from classifier_jev.jev_client import JevClient
from classifier_jev.jev_classifier import JevClassifier
from classifier_jev.questions import build_questions
from dataclasses import asdict


async def classify_ticket(ticket, model="gpt-6-luna", prompt="v7"):
    if not ticket.strip():
        raise ValueError("Ticket must not be empty")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY in .env or the environment")
    settings = load_config("settings.yaml")
    prices = load_config("prices.yaml")
    async with AsyncExitStack() as stack:
        client = await stack.enter_async_context(
            httpx.AsyncClient(timeout=settings["failure"]["timeout_seconds"])
        )
        if model == "jev-1.13.0":
            config = yaml.safe_load(
                (ROOT / "02/config/jev.yaml").read_text(encoding="utf-8")
            )
            questions = build_questions(config["questions_version"])
            spec = {
                "id": model,
                "slug": config["slug"],
                "params": {},
                "price_key": model,
            }
            provider = JevClient(client, config, key, questions)
            classifier = JevClassifier(
                [spec],
                {model: provider},
                settings,
                prices,
                prompt_version=config["questions_version"],
                prompt_hash=hash_object([asdict(q) for q in questions]),
                jev_config=config,
            )
        else:
            config = load_config("models.yaml")
            if (
                model not in config["models"]
                or config["models"][model]["provider"] == "mock"
            ):
                raise ValueError("Choose a configured live model")
            text, digest = load_prompt(prompt)
            classifier = LLMClassifier(
                [config["models"][model]],
                {model: OpenRouterProvider(client, config["base_url"], key)},
                settings,
                prices,
                prompt,
                text,
                digest,
            )
        result = await classifier.classify(ticket)
    # Queue format reuses Case; this machine-derived proposal is not a gold label.
    proposal = Expected(
        primary_category=result.primary_category,
        secondary_categories=result.secondary_categories,
        priority_markers=result.priority_markers,
        action_markers=result.action_markers,
        priority=result.priority,
        next_step=result.next_step_raw,
    )
    case = Case(
        id=0,
        difficulty="hard",
        language=result.language or "unknown",
        ticket=ticket,
        expected=proposal,
    )
    append_review(ROOT / "01/review_queue.jsonl", case, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--file", type=Path)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--prompt", default="v7")
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()
    if not args.allow_live:
        parser.error("Add --allow-live to authorize a paid model request")
    try:
        ticket = (
            args.text
            if args.text is not None
            else args.file.read_text(encoding="utf-8")
        )
        result = asyncio.run(classify_ticket(ticket, args.model, args.prompt))
        print(
            json.dumps(
                result.model_dump(exclude={"raw_response", "attempts"}),
                ensure_ascii=False,
                indent=2,
            )
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
