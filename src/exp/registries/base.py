# Copyright (c) 2026 "Sheaf Neural Networks as Message Passing"
# Authors: Alessio Borgi, Luke Braithwaite, Mario Severino, Emanuele Mule,
#   Fabrizio Silvestri, and Pietro Liò

"""Generic registry base class."""


class Registry[Key, Value]:
    """A typed key-value store that raises on duplicate keys and missing lookups."""

    def __init__(self) -> None:
        self._store: dict[Key, Value] = {}

    def register(self, key: Key, value: Value) -> None:
        if key in self._store:
            raise ValueError(f"Key {key!r} is already registered.")
        self._store[key] = value

    def get(self, key: Key) -> Value:
        if key not in self._store:
            raise KeyError(f"Key {key!r} not found in registry.")
        return self._store[key]

    def __contains__(self, key: object) -> bool:  # noqa: D105
        return key in self._store

    def list_keys(self) -> list[Key]:
        return list(self._store.keys())
