"""Create once; hash-pin the split and reject later changes."""

import hashlib
import json
import random
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_split(data, cases):
    by_id = {c.id: c for c in cases}
    dev, holdout = set(data["dev"]), set(data["holdout"])
    assert len(dev) == 40 and len(holdout) == 30 and not dev & holdout
    assert dev | holdout == set(by_id)
    assert {
        d: sum(by_id[i].difficulty == d for i in holdout)
        for d in ("easy", "medium", "hard")
    } == {"easy": 9, "medium": 11, "hard": 10}
    for attr in ("language", "category"):
        value = lambda c: (
            c.language if attr == "language" else c.expected.primary_category
        )
        for label in {value(c) for c in cases}:
            ids = {c.id for c in cases if value(c) == label}
            if attr == "category" or len(ids) >= 2:
                assert ids & dev and ids & holdout
    assert (50 in dev) != (65 in dev)


def get_split(path, cases):
    path = Path(path)
    lock = path.with_suffix(".sha256")
    if path.exists():
        if (
            not lock.exists()
            or digest(path) != lock.read_text(encoding="utf-8").strip()
        ):
            raise ValueError("Split hash changed or missing; refusing to regenerate")
        data = json.loads(path.read_text(encoding="utf-8"))
        check_split(data, cases)
        return data
    if lock.exists():
        raise ValueError("Split is missing but its hash exists")
    for seed in range(42, 100042):
        rng = random.Random(seed)
        holdout = []
        for difficulty, n in [("easy", 9), ("medium", 11), ("hard", 10)]:
            holdout.extend(
                rng.sample(sorted(c.id for c in cases if c.difficulty == difficulty), n)
            )
        data = {
            "seed": seed,
            "dev": sorted({c.id for c in cases} - set(holdout)),
            "holdout": sorted(holdout),
        }
        try:
            check_split(data, cases)
        except AssertionError:
            continue
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        lock.write_text(digest(path) + "\n", encoding="utf-8")
        return data
    raise ValueError("Could not satisfy split constraints")
