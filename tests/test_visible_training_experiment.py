from pathlib import Path
from types import SimpleNamespace

from scripts.evaluate_agent import (
    OpponentEvaluationResult,
    PolicyEpisodeResult,
    SavedPolicyEvaluationResult,
)
from scripts.evaluate_single_agent import EpisodeResult, EvaluationResult
from scripts.run_training_experiment import (
    ExperimentComparisonRow,
    VisibleExperimentResult,
    build_comparison_rows,
    format_comparison_table,
    log_comparison_table,
    make_jsonable,
    run_training_experiment,
    validate_experiment_args,
)
from scripts.train_single_agent import TrainingResult

import pytest


class FakeRun:
    def __init__(self) -> None:
        self.logs = []
        self.finished = False

    def log(self, data) -> None:
        self.logs.append(data)

    def finish(self) -> None:
        self.finished = True


def make_policy_result(opponent: str, win_rate: float, reward: float) -> OpponentEvaluationResult:
    episode = PolicyEpisodeResult(
        episode=0,
        seed=1,
        opponent_strategy=opponent,
        reward=reward,
        win=win_rate > 0.0,
        turns_survived=10,
        truncated=False,
        winner="player_1" if win_rate > 0.0 else "player_2",
        learner_actions=5,
        defuses=0,
        explosions=1,
        cards_played={},
    )
    return OpponentEvaluationResult(
        opponent_strategy=opponent,
        episodes=1,
        wins=int(win_rate),
        win_rate=win_rate,
        average_reward=reward,
        average_turns_survived=10.0,
        truncations=0,
        truncation_rate=0.0,
        defuses=0,
        explosions=1,
        cards_played={},
        episode_results=(episode,),
    )


def make_random_result(win_rate: float, reward: float) -> EvaluationResult:
    episode = EpisodeResult(
        episode=0,
        seed=1,
        reward=reward,
        win=win_rate > 0.0,
        turns=12,
        truncated=False,
        winner="player_1" if win_rate > 0.0 else "player_2",
        learner_actions=6,
    )
    return EvaluationResult(
        episodes=1,
        wins=int(win_rate),
        win_rate=win_rate,
        average_reward=reward,
        average_turns=12.0,
        truncations=0,
        truncation_rate=0.0,
        episode_results=(episode,),
        trace_rows=(),
    )


def test_build_comparison_rows_calculates_deltas() -> None:
    trained = SavedPolicyEvaluationResult(
        model_path=Path("models/model.zip"),
        learner="player_1",
        players=2,
        opponent_results=(make_policy_result("draw-only", 0.75, 0.5),),
    )
    random_results = {"draw-only": make_random_result(0.25, -0.5)}

    rows = build_comparison_rows(trained, random_results)

    assert rows == (
        ExperimentComparisonRow(
            opponent_strategy="draw-only",
            trained_win_rate=0.75,
            random_win_rate=0.25,
            win_rate_delta=0.5,
            trained_average_reward=0.5,
            random_average_reward=-0.5,
            reward_delta=1.0,
            trained_average_turns=10.0,
            random_average_turns=12.0,
        ),
    )


def test_run_training_experiment_writes_report_with_monkeypatched_pipeline(
    monkeypatch,
    tmp_path: Path,
) -> None:
    training = TrainingResult(
        model_path=tmp_path / "model.zip",
        total_timesteps=16,
        eval_episodes=1,
        eval_wins=1,
        eval_win_rate=1.0,
        eval_average_reward=1.0,
        eval_average_turns=10.0,
    )
    trained = SavedPolicyEvaluationResult(
        model_path=training.model_path,
        learner="player_1",
        players=2,
        opponent_results=(make_policy_result("draw-only", 0.5, 0.0),),
    )

    monkeypatch.setattr(
        "scripts.run_training_experiment.train_single_agent",
        lambda **kwargs: training,
    )
    monkeypatch.setattr(
        "scripts.run_training_experiment.evaluate_saved_policy",
        lambda **kwargs: trained,
    )
    monkeypatch.setattr(
        "scripts.run_training_experiment.evaluate_single_agent",
        lambda **kwargs: make_random_result(0.25, -0.5),
    )

    result = run_training_experiment(
        total_timesteps=16,
        episodes=1,
        opponent_strategies=("draw-only",),
        model_dir=tmp_path,
        report_dir=tmp_path / "reports",
        run_name="test_experiment",
        n_steps=8,
        batch_size=4,
        n_epochs=1,
    )

    assert result.model_path == training.model_path
    assert result.report_path.exists()
    assert result.comparison_rows[0].win_rate_delta == 0.25


def test_format_comparison_table_mentions_delta() -> None:
    table = format_comparison_table(
        (
            ExperimentComparisonRow(
                opponent_strategy="random",
                trained_win_rate=0.4,
                random_win_rate=0.2,
                win_rate_delta=0.2,
                trained_average_reward=-0.2,
                random_average_reward=-0.6,
                reward_delta=0.4,
                trained_average_turns=30.0,
                random_average_turns=25.0,
            ),
        )
    )

    assert "random" in table
    assert "+0.200" in table


def test_log_comparison_table_uses_wandb_table(monkeypatch) -> None:
    class FakeTable:
        def __init__(self, columns) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *row) -> None:
            self.rows.append(row)

    fake_wandb = SimpleNamespace(Table=FakeTable)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    result = VisibleExperimentResult(
        run_name="test",
        model_path=Path("models/model.zip"),
        report_path=Path("reports/test.json"),
        training=TrainingResult(
            model_path=Path("models/model.zip"),
            total_timesteps=16,
            eval_episodes=1,
            eval_wins=1,
            eval_win_rate=1.0,
            eval_average_reward=1.0,
            eval_average_turns=10.0,
        ),
        comparison_rows=(
            ExperimentComparisonRow(
                opponent_strategy="draw-only",
                trained_win_rate=0.5,
                random_win_rate=0.25,
                win_rate_delta=0.25,
                trained_average_reward=0.0,
                random_average_reward=-0.5,
                reward_delta=0.5,
                trained_average_turns=10.0,
                random_average_turns=12.0,
            ),
        ),
    )
    run = FakeRun()

    log_comparison_table(run, result)

    table = run.logs[-1]["comparison/trained_vs_random"]
    assert table.rows
    assert table.rows[0][0] == "draw-only"


def test_make_jsonable_converts_paths_and_tuples() -> None:
    assert make_jsonable({"path": Path("models/x.zip"), "items": ("a", "b")}) == {
        "path": "models\\x.zip" if "\\" in str(Path("models/x.zip")) else "models/x.zip",
        "items": ["a", "b"],
    }


def test_validate_experiment_args_rejects_empty_opponents() -> None:
    with pytest.raises(ValueError, match="At least one opponent"):
        validate_experiment_args(
            total_timesteps=16,
            episodes=1,
            opponent_strategies=(),
            wandb_mode="disabled",
        )
