from typing import Protocol


class MemoryService(Protocol):
    def load(self, conversation_id: str) -> dict:
        ...

    def save(self, conversation_id: str, state: dict) -> None:
        ...


class InMemoryMemoryService:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def load(self, conversation_id: str) -> dict:
        return self._store.get(conversation_id, {})

    def save(self, conversation_id: str, state: dict) -> None:
        self._store[conversation_id] = state
