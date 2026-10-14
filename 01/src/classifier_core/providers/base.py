from dataclasses import dataclass, field
from typing import Protocol
from ..schema import Usage


@dataclass
class Response:
    text: str
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None
    metadata: dict = field(default_factory=dict)


class ProviderError(Exception):
    """Only safe error codes leave adapters; never serialize request headers/errors."""

    def __init__(self, kind, retry_after=None, response=None, validation_error=None):
        super().__init__(kind)
        self.kind, self.retry_after, self.response = kind, retry_after, response
        self.validation_error = validation_error


class Provider(Protocol):
    async def complete(
        self,
        *,
        ticket: str,
        model: dict,
        system: str,
        user: str,
        repair: dict | None = None,
    ) -> Response: ...
