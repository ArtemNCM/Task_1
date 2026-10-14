import hashlib
import json
import sqlite3
import unicodedata
from .schema import ClassifierResult


def cache_key(ticket, model_id, model_params, prompt_hash, taxonomy_version):
    # Preserve internal whitespace/case: either may matter for interpretation.
    normalized = unicodedata.normalize("NFC", ticket).replace("\r\n", "\n").strip()
    payload = [normalized, model_id, model_params, prompt_hash, taxonomy_version]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class Cache:
    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    def get(self, key):
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT value FROM results WHERE key=?", (key,)).fetchone()
        if not row:
            return None
        result = ClassifierResult.model_validate_json(row[0])
        result.cache_hit = True
        # Cache is a new delivery with no new billing or transport attempts.
        result.attempts = []
        result.cost_usd = 0.0
        result.latency_ms = 0.0
        from .schema import Usage

        result.usage = Usage(
            input_tokens=0, output_tokens=0, cached_input_tokens=0, reasoning_tokens=0
        )
        return result

    def put(self, key, result):
        if result.status not in ("ok", "recovered", "repaired"):
            return
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT OR REPLACE INTO results VALUES (?,?)",
                (key, result.model_dump_json()),
            )

    def clear(self):
        with sqlite3.connect(self.path) as db:
            db.execute("DELETE FROM results")
