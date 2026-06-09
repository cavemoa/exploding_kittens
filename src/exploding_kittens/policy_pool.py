"""Policy-pool helpers for self-play experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any


SCRIPTED_POLICY_NAMES = ("random", "safe-rule", "draw-only")
POLICY_POOL_SAMPLE_MODES = ("latest-only", "random-historical", "mixed")


@dataclass(frozen=True)
class PolicyPoolEntry:
    name: str
    policy_spec: str
    kind: str
    role: str
    checkpoint_path: Path | None = None
    iteration: int | None = None
    score: float | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "policy_spec": self.policy_spec,
            "kind": self.kind,
            "role": self.role,
            "checkpoint_path": (
                str(self.checkpoint_path) if self.checkpoint_path is not None else None
            ),
            "iteration": self.iteration,
            "score": self.score,
        }


@dataclass(frozen=True)
class PolicyPool:
    entries: tuple[PolicyPoolEntry, ...]

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Path,
        *,
        scores: dict[str, float] | None = None,
        include_scripted: bool = True,
    ) -> "PolicyPool":
        entries: list[PolicyPoolEntry] = []
        if include_scripted:
            entries.extend(scripted_entries())
        checkpoints = discover_rllib_checkpoints(checkpoint_dir)
        entries.extend(checkpoint_entries(checkpoints, scores=scores or {}))
        return cls(tuple(entries))

    @property
    def scripted_entries(self) -> tuple[PolicyPoolEntry, ...]:
        return tuple(entry for entry in self.entries if entry.kind == "scripted")

    @property
    def historical_entries(self) -> tuple[PolicyPoolEntry, ...]:
        return tuple(entry for entry in self.entries if entry.kind == "rllib")

    def entry_by_role(self, role: str) -> PolicyPoolEntry | None:
        for entry in self.entries:
            if entry.role == role:
                return entry
        return None

    def sample_opponents(
        self,
        *,
        players: int,
        mode: str,
        rng: random.Random,
        learner_seat: str = "player_1",
    ) -> dict[str, str]:
        if players < 2:
            raise ValueError("players must be at least 2.")
        if mode not in POLICY_POOL_SAMPLE_MODES:
            raise ValueError(
                f"mode must be one of: {', '.join(POLICY_POOL_SAMPLE_MODES)}."
            )

        opponent_seats = tuple(
            f"player_{index}"
            for index in range(1, players + 1)
            if f"player_{index}" != learner_seat
        )
        if mode == "latest-only":
            latest = self.entry_by_role("latest")
            if latest is None:
                raise ValueError("latest-only mode requires a latest checkpoint entry.")
            return {seat: latest.policy_spec for seat in opponent_seats}

        if mode == "random-historical":
            candidates = self.historical_entries
            if not candidates:
                raise ValueError(
                    "random-historical mode requires at least one RLlib checkpoint entry."
                )
            return {
                seat: rng.choice(candidates).policy_spec
                for seat in opponent_seats
            }

        candidates = (*self.scripted_entries, *self.historical_entries)
        if not candidates:
            raise ValueError("mixed mode requires at least one policy-pool entry.")
        return {
            seat: rng.choice(candidates).policy_spec
            for seat in opponent_seats
        }

    def save_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"entries": [entry.to_metadata() for entry in self.entries]}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scripted_entries() -> tuple[PolicyPoolEntry, ...]:
    return tuple(
        PolicyPoolEntry(
            name=name,
            policy_spec=name,
            kind="scripted",
            role="scripted",
        )
        for name in SCRIPTED_POLICY_NAMES
    )


def discover_rllib_checkpoints(checkpoint_dir: Path) -> tuple[Path, ...]:
    if not checkpoint_dir.exists():
        return ()
    candidates = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and (path / "rllib_checkpoint.json").exists()
    ]
    if (checkpoint_dir / "rllib_checkpoint.json").exists():
        candidates.append(checkpoint_dir)
    return tuple(sorted(set(candidates), key=checkpoint_sort_key))


def checkpoint_entries(
    checkpoints: tuple[Path, ...],
    *,
    scores: dict[str, float],
) -> tuple[PolicyPoolEntry, ...]:
    if not checkpoints:
        return ()

    latest = checkpoints[-1]
    previous = checkpoints[-2] if len(checkpoints) > 1 else None
    best = best_checkpoint(checkpoints, scores) or latest
    entries: list[PolicyPoolEntry] = []
    for checkpoint in checkpoints:
        roles = checkpoint_roles(
            checkpoint,
            latest=latest,
            previous=previous,
            best=best,
        )
        for role in roles:
            entries.append(
                PolicyPoolEntry(
                    name=f"{role}:{checkpoint.name}",
                    policy_spec=f"rllib:{checkpoint}",
                    kind="rllib",
                    role=role,
                    checkpoint_path=checkpoint,
                    iteration=checkpoint_iteration(checkpoint),
                    score=checkpoint_score(checkpoint, scores),
                )
            )
    return tuple(entries)


def checkpoint_roles(
    checkpoint: Path,
    *,
    latest: Path,
    previous: Path | None,
    best: Path,
) -> tuple[str, ...]:
    roles: list[str] = ["checkpoint"]
    if checkpoint == latest:
        roles.append("latest")
    if previous is not None and checkpoint == previous:
        roles.append("previous")
    if checkpoint == best:
        roles.append("best")
    return tuple(roles)


def best_checkpoint(
    checkpoints: tuple[Path, ...],
    scores: dict[str, float],
) -> Path | None:
    scored = [
        (checkpoint_score(checkpoint, scores), checkpoint)
        for checkpoint in checkpoints
        if checkpoint_score(checkpoint, scores) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda item: item[0])[1]


def checkpoint_score(checkpoint: Path, scores: dict[str, float]) -> float | None:
    for key in (str(checkpoint), checkpoint.name):
        if key in scores:
            return float(scores[key])
    return None


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    iteration = checkpoint_iteration(path)
    if iteration is None:
        return (10**12, path.name)
    return (iteration, path.name)


def checkpoint_iteration(path: Path) -> int | None:
    prefix = "checkpoint_"
    if not path.name.startswith(prefix):
        return None
    try:
        return int(path.name.removeprefix(prefix))
    except ValueError:
        return None
