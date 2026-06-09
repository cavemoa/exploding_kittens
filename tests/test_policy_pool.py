import json
import random
from pathlib import Path

import pytest

from exploding_kittens.policy_pool import (
    PolicyPool,
    best_checkpoint,
    checkpoint_iteration,
    checkpoint_score,
    discover_rllib_checkpoints,
    scripted_entries,
)
from scripts.build_policy_pool import build_policy_pool, load_scores


def test_scripted_entries_include_baselines() -> None:
    entries = scripted_entries()

    assert {entry.policy_spec for entry in entries} == {
        "random",
        "safe-rule",
        "draw-only",
    }
    assert all(entry.kind == "scripted" for entry in entries)


def test_discover_rllib_checkpoints_sorts_iteration_dirs_and_final_root(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    create_checkpoint(run_dir / "checkpoint_000002")
    create_checkpoint(run_dir / "checkpoint_000001")
    create_checkpoint(run_dir)

    checkpoints = discover_rllib_checkpoints(run_dir)

    assert [checkpoint.name for checkpoint in checkpoints] == [
        "checkpoint_000001",
        "checkpoint_000002",
        "run",
    ]


def test_policy_pool_adds_latest_previous_and_best_checkpoint_entries(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    create_checkpoint(run_dir / "checkpoint_000001")
    create_checkpoint(run_dir / "checkpoint_000002")
    create_checkpoint(run_dir / "checkpoint_000003")
    scores = {"checkpoint_000002": 0.75, "checkpoint_000003": 0.25}

    pool = PolicyPool.from_checkpoint_dir(run_dir, scores=scores)

    latest = pool.entry_by_role("latest")
    previous = pool.entry_by_role("previous")
    best = pool.entry_by_role("best")
    assert latest is not None
    assert previous is not None
    assert best is not None
    assert latest.checkpoint_path.name == "checkpoint_000003"
    assert previous.checkpoint_path.name == "checkpoint_000002"
    assert best.checkpoint_path.name == "checkpoint_000002"
    assert best.score == 0.75


def test_sample_opponents_supports_latest_random_historical_and_mixed(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    create_checkpoint(run_dir / "checkpoint_000001")
    create_checkpoint(run_dir / "checkpoint_000002")
    pool = PolicyPool.from_checkpoint_dir(run_dir)

    latest = pool.sample_opponents(
        players=3,
        mode="latest-only",
        rng=random.Random(1),
    )
    historical = pool.sample_opponents(
        players=3,
        mode="random-historical",
        rng=random.Random(1),
    )
    mixed = pool.sample_opponents(
        players=3,
        mode="mixed",
        rng=random.Random(2),
    )

    assert set(latest) == {"player_2", "player_3"}
    assert all(value.startswith("rllib:") for value in latest.values())
    assert all(value.startswith("rllib:") for value in historical.values())
    assert set(mixed) == {"player_2", "player_3"}
    assert set(mixed.values()) <= {
        *(entry.policy_spec for entry in pool.scripted_entries),
        *(entry.policy_spec for entry in pool.historical_entries),
    }


def test_sample_opponents_rejects_missing_checkpoint_for_latest_only() -> None:
    pool = PolicyPool(scripted_entries())

    with pytest.raises(ValueError, match="latest checkpoint"):
        pool.sample_opponents(players=3, mode="latest-only", rng=random.Random(1))


def test_save_metadata_writes_entries(tmp_path: Path) -> None:
    pool = PolicyPool(scripted_entries())
    metadata_path = tmp_path / "reports" / "policy_pool.json"

    pool.save_metadata(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(payload["entries"]) == 3
    assert payload["entries"][0]["kind"] == "scripted"


def test_best_checkpoint_and_score_helpers(tmp_path: Path) -> None:
    first = tmp_path / "checkpoint_000001"
    second = tmp_path / "checkpoint_000002"

    assert checkpoint_iteration(first) == 1
    assert checkpoint_score(first, {str(first): 0.1}) == 0.1
    assert best_checkpoint((first, second), {second.name: 0.9}) == second


def test_build_policy_pool_script_helper_saves_metadata_and_samples(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    metadata_path = tmp_path / "reports" / "pool.json"
    create_checkpoint(run_dir / "checkpoint_000001")

    sampled = build_policy_pool(
        checkpoint_dir=run_dir,
        metadata_path=metadata_path,
        players=3,
        sample_mode="latest-only",
        seed=1,
        scores={},
    )

    assert metadata_path.exists()
    assert sampled == {
        "player_2": f"rllib:{run_dir / 'checkpoint_000001'}",
        "player_3": f"rllib:{run_dir / 'checkpoint_000001'}",
    }


def test_load_scores_reads_json_object(tmp_path: Path) -> None:
    scores_path = tmp_path / "scores.json"
    scores_path.write_text('{"checkpoint_000001": 0.5}', encoding="utf-8")

    assert load_scores(scores_path) == {"checkpoint_000001": 0.5}


def create_checkpoint(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "rllib_checkpoint.json").write_text("{}", encoding="utf-8")
