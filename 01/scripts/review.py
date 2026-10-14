import json
from classifier_core.taxonomy import ROOT
from classifier_core.schema import Decision, ClassifierResult
from classifier_core.rules import derive
from classifier_core.hitl import save_correction


def main():
    queue = ROOT / "review_queue.jsonl"
    corrections = ROOT / "data/corrections.jsonl"
    done = (
        {
            json.loads(line)["queue_id"]
            for line in corrections.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if corrections.exists()
        else set()
    )
    pending = (
        [
            json.loads(line)
            for line in queue.read_text(encoding="utf-8").splitlines()
            if line.strip() and json.loads(line)["queue_id"] not in done
        ]
        if queue.exists()
        else []
    )
    if not pending:
        print("No pending review items")
        return
    item = pending[0]
    m = ClassifierResult.model_validate(item["machine"])
    print(item["ticket"])
    print(
        json.dumps(
            m.model_dump(exclude={"attempts", "raw_response"}),
            ensure_ascii=False,
            indent=2,
        )
    )
    action = (
        input("Accept machine classification (a), correct (c), or quit (q): ")
        .strip()
        .lower()
    )
    if action not in ("a", "c"):
        return
    d = Decision.model_validate(m.model_dump(include=set(Decision.model_fields)))
    if action == "c":
        category = input("Primary category [keep]: ").strip()
        pm = input("Priority markers comma-separated [keep; - clears]: ").strip()
        am = input("Action markers [keep; - clears]: ").strip()
        secondary = input("Secondary categories [keep; - clears]: ").strip()
        values = d.model_dump()
        if category:
            values["primary_category"] = category
        for key, value in [
            ("priority_markers", pm),
            ("action_markers", am),
            ("secondary_categories", secondary),
        ]:
            if value:
                values[key] = (
                    [] if value == "-" else [v.strip() for v in value.split(",")]
                )
        hint = input("Routing hint [keep; marketing; - clears]: ").strip()
        if hint:
            values["routing_hint"] = None if hint == "-" else hint
        if values["primary_category"] != "feedback":
            values["routing_hint"] = None
        d = Decision.model_validate(values)
    derived = derive(d)
    step = (
        input(f"Next step [{derived['next_step_raw']}]: ").strip()
        or derived["next_step_raw"]
    )
    expected = d.model_dump(
        include={
            "primary_category",
            "secondary_categories",
            "priority_markers",
            "action_markers",
        }
    ) | {"priority": derived["priority"], "next_step": step}
    save_correction(corrections, item, expected, input("Reviewer note (required): "))
    print("Correction appended; original gold dataset unchanged.")


if __name__ == "__main__":
    main()
