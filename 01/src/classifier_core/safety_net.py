import re
from .taxonomy import load_config


def detect(ticket):
    return [
        marker
        for marker, languages in load_config("safety_keywords.yaml").items()
        if any(
            re.search(pattern, ticket, re.I)
            for patterns in languages.values()
            for pattern in patterns
        )
    ]
