"""
BackendResolver: selects the best ExecutionBackend for a given ExecutionPolicy.

Resolution strategy:
1. Filter backends by availability, runtime support, and isolation support.
2. Sort remaining candidates by isolation strength (stronger is better).
3. Return the best match.
"""

from __future__ import annotations

from llamagent.modules.sandbox.backend import ExecutionBackend
from llamagent.modules.sandbox.policy import ExecutionPolicy

# Isolation levels ordered by strength (weakest to strongest).
_ISOLATION_STRENGTH: dict[str, int] = {
    "none": 0,
    "process": 1,
    "container": 2,
    "microvm": 3,
}


class BackendResolver:
    """Registry + resolver that picks the best backend for a policy."""

    def __init__(self) -> None:
        self._backends: list[ExecutionBackend] = []

    def register(self, backend: ExecutionBackend) -> None:
        """Register a backend. Later registrations take priority on ties."""
        self._backends.append(backend)

    def resolve(self, policy: ExecutionPolicy) -> ExecutionBackend:
        """
        Find the best backend that can satisfy *policy*.

        Filtering rules:
        - Backend must be available.
        - Backend must support the requested runtime.
        - Backend must support the requested isolation level (or stronger).

        Among remaining candidates the one with the strongest isolation is
        returned.  Raises RuntimeError if no backend matches.
        """
        required_strength = _ISOLATION_STRENGTH.get(policy.isolation, 0)
        candidates: list[tuple[int, ExecutionBackend]] = []

        for backend in self._backends:
            caps = backend.capabilities()

            if not caps.get("available", False):
                continue

            if policy.runtime not in caps.get("supported_runtimes", []):
                continue

            # Check that the backend can provide *at least* the requested
            # isolation level.
            supported = caps.get("supported_isolation", [])
            max_strength = max(
                (_ISOLATION_STRENGTH.get(level, 0) for level in supported),
                default=-1,
            )
            if max_strength < required_strength:
                continue

            # Check network isolation capability if policy requires it.
            if policy.network != "full" and not caps.get("supports_network_isolation", False):
                continue

            candidates.append((max_strength, backend))

        if not candidates:
            raise RuntimeError(
                f"No sandbox backend available for policy: "
                f"runtime={policy.runtime!r}, isolation={policy.isolation!r}. "
                f"Registered backends: {[b.name for b in self._backends]}"
            )

        # Sort by isolation strength descending; last-registered wins ties.
        candidates.sort(key=lambda pair: pair[0], reverse=True)
        return candidates[0][1]
