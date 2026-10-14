import hashlib
import json
from datetime import datetime, timezone
from .schema import Case, Expected


def queue_id(ticket, result):
    content = {
        "ticket": ticket,
        "model": result.model_id,
        "prompt": result.prompt_version,
        "category": result.primary_category,
        "markers": result.priority_markers,
        "actions": result.action_markers,
        "next_step": result.next_step_final,
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def append_review(path, case, result):
    if not (result.needs_review or result.automation_level == "human_required"):
        return
    uid = queue_id(case.ticket, result)
    if path.exists() and any(
        json.loads(line)["queue_id"] == uid
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ):
        return
    row = {
        "queue_id": uid,
        "case": case.model_dump(),
        "ticket": case.ticket,
        "machine": result.model_dump(),
        "reason": result.review_reasons or ["human_required"],
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_correction(path, item, expected, reviewer_note):
    if not reviewer_note.strip():
        raise ValueError("Reviewer note required")
    # Distinct feedback IDs; stable for a queue item and never overwrite gold IDs.
    correction_id = 1_000_000 + int(item["queue_id"][:12], 16)
    human = Case(
        id=correction_id,
        difficulty=item["case"]["difficulty"],
        language=item["case"]["language"],
        ticket=item["ticket"],
        expected=Expected.model_validate(expected),
    )
    row = {
        "queue_id": item["queue_id"],
        "ticket": item["ticket"],
        "machine": item["machine"],
        "human": human.model_dump(),
        "reviewer_note": reviewer_note,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
