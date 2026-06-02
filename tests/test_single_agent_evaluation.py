from types import SimpleNamespace

import pytest

from scripts.evaluate_single_agent import (
    TRACE_COLUMNS,
    evaluate_single_agent,
    log_summary,
    log_trace_table,
    start_wandb_run,
)


class FakeRun:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, data, step=None) -> None:
        self.logs.append((data, step))

    def finish(self) -> None:
        self.finished = True


def test_evaluate_single_agent_runs_without_wandb_dependency() -> None:
    result = evaluate_single_agent(
        episodes=3,
        players=2,
        opponent_strategy="draw-only",
        seed=1,
        max_turns=500,
        trace_episodes=1,
        wandb_mode="disabled",
    )

    assert result.episodes == 3
    assert len(result.episode_results) == 3
    assert 0.0 <= result.win_rate <= 1.0
    assert result.average_turns > 0
    assert result.truncation_rate == 0.0
    assert result.trace_rows
    assert all(len(row) == len(TRACE_COLUMNS) for row in result.trace_rows)


def test_evaluate_single_agent_is_deterministic_for_fixed_seed() -> None:
    first = evaluate_single_agent(
        episodes=3,
        players=3,
        opponent_strategy="safe-rule",
        seed=7,
        max_turns=500,
        trace_episodes=2,
        wandb_mode="disabled",
    )
    second = evaluate_single_agent(
        episodes=3,
        players=3,
        opponent_strategy="safe-rule",
        seed=7,
        max_turns=500,
        trace_episodes=2,
        wandb_mode="disabled",
    )

    assert first == second


def test_start_wandb_run_disabled_returns_none() -> None:
    run = start_wandb_run(
        mode="disabled",
        project="test-project",
        entity=None,
        config={"episodes": 1},
    )

    assert run is None


def test_evaluate_single_agent_rejects_unknown_wandb_mode() -> None:
    with pytest.raises(ValueError, match="wandb_mode"):
        evaluate_single_agent(wandb_mode="quiet")


def test_log_summary_sends_aggregate_metrics() -> None:
    run = FakeRun()
    result = evaluate_single_agent(
        episodes=2,
        players=2,
        opponent_strategy="draw-only",
        seed=4,
        max_turns=500,
        trace_episodes=0,
        wandb_mode="disabled",
    )

    log_summary(run, result)

    assert run.logs
    assert "eval/win_rate" in run.logs[-1][0]
    assert "eval/average_reward" in run.logs[-1][0]


def test_log_trace_table_uses_wandb_table(monkeypatch) -> None:
    class FakeTable:
        def __init__(self, columns) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *row) -> None:
            self.rows.append(row)

    run = FakeRun()
    fake_wandb = SimpleNamespace(Table=FakeTable)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    row = tuple(range(len(TRACE_COLUMNS)))
    log_trace_table(run, (row,))

    logged_table = run.logs[-1][0]["traces/episodes"]
    assert logged_table.columns == list(TRACE_COLUMNS)
    assert logged_table.rows == [row]
