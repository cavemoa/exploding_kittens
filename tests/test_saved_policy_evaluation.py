from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_agent import (
    SavedPolicyEvaluationResult,
    evaluate_policy_against_opponent,
    format_results_table,
    log_comparison_table,
    log_results,
    parse_name_list,
    start_wandb_run,
    summarize_opponent_results,
    validate_evaluation_args,
)


class FirstLegalActionModel:
    def predict(
        self,
        observation,
        *,
        deterministic=True,
        action_masks=None,
    ):
        del observation, deterministic
        return int(np.flatnonzero(action_masks)[0]), None


class FakeRun:
    def __init__(self) -> None:
        self.logs = []

    def log(self, data) -> None:
        self.logs.append(data)


def test_evaluate_policy_against_opponent_runs_fake_saved_policy() -> None:
    result = evaluate_policy_against_opponent(
        FirstLegalActionModel(),
        episodes=2,
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=1,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        deterministic=True,
    )

    assert result.opponent_strategy == "draw-only"
    assert result.episodes == 2
    assert 0.0 <= result.win_rate <= 1.0
    assert result.average_turns_survived > 0
    assert result.explosions >= 1


def test_summarize_opponent_results_aggregates_episode_metrics() -> None:
    result = evaluate_policy_against_opponent(
        FirstLegalActionModel(),
        episodes=2,
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=3,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        deterministic=True,
    )

    summary = summarize_opponent_results(
        "draw-only",
        list(result.episode_results),
    )

    assert summary == result


def test_validate_evaluation_args_rejects_missing_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Model file does not exist"):
        validate_evaluation_args(
            model_path=tmp_path / "missing.zip",
            episodes=1,
            opponent_strategies=("draw-only",),
            wandb_mode="disabled",
        )


def test_start_wandb_run_disabled_returns_none() -> None:
    assert (
        start_wandb_run(
            mode="disabled",
            project="test-project",
            entity=None,
            config={},
        )
        is None
    )


def test_log_results_sends_strategy_metrics_and_table(monkeypatch) -> None:
    class FakeTable:
        def __init__(self, columns) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *row) -> None:
            self.rows.append(row)

    fake_wandb = SimpleNamespace(Table=FakeTable)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)

    opponent_result = evaluate_policy_against_opponent(
        FirstLegalActionModel(),
        episodes=1,
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=4,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        deterministic=True,
    )
    result = SavedPolicyEvaluationResult(
        model_path=Path("models/test.zip"),
        learner="player_1",
        players=2,
        opponent_results=(opponent_result,),
    )
    run = FakeRun()

    log_results(run, result)

    assert any("eval/draw-only/win_rate" in log for log in run.logs)
    assert "eval/opponent_comparison" in run.logs[-1]


def test_log_comparison_table_ignores_disabled_run() -> None:
    opponent_result = evaluate_policy_against_opponent(
        FirstLegalActionModel(),
        episodes=1,
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=5,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        deterministic=True,
    )
    result = SavedPolicyEvaluationResult(
        model_path=Path("models/test.zip"),
        learner="player_1",
        players=2,
        opponent_results=(opponent_result,),
    )

    assert log_comparison_table(None, result) is None


def test_format_results_table_includes_opponent_names() -> None:
    opponent_result = evaluate_policy_against_opponent(
        FirstLegalActionModel(),
        episodes=1,
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=6,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        deterministic=True,
    )
    table = format_results_table(
        SavedPolicyEvaluationResult(
            model_path=Path("models/test.zip"),
            learner="player_1",
            players=2,
            opponent_results=(opponent_result,),
        )
    )

    assert "draw-only" in table
    assert "win_rate" in table


def test_parse_name_list_ignores_empty_items() -> None:
    assert parse_name_list("draw-only, random,, safe-rule ") == (
        "draw-only",
        "random",
        "safe-rule",
    )
