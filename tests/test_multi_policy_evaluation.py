from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_multi_policy_game import (
    ModelController,
    MultiPolicyEvaluationResult,
    RandomMaskController,
    SeatEvaluationResult,
    build_controller,
    complete_policy_specs,
    evaluate_multi_policy_game,
    format_results_table,
    log_results_table,
    parse_seat_policies,
    run_multi_policy_episode,
    summarize_evaluation,
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


def test_parse_seat_policies_accepts_models_and_scripted_policies() -> None:
    policies = parse_seat_policies(
        "player_1=model:models/a.zip, player_2=model:models/b.zip, player_3=random"
    )

    assert policies == {
        "player_1": "model:models/a.zip",
        "player_2": "model:models/b.zip",
        "player_3": "random",
    }


def test_complete_policy_specs_defaults_missing_seats_to_random() -> None:
    specs = complete_policy_specs(
        {"player_1": "safe-rule"},
        ("player_1", "player_2", "player_3"),
    )

    assert specs["player_1"].policy == "safe-rule"
    assert specs["player_2"].policy == "random"
    assert specs["player_3"].policy == "random"


def test_build_controller_rejects_missing_model() -> None:
    specs = complete_policy_specs(
        {"player_1": "model:models/missing.zip"},
        ("player_1", "player_2"),
    )

    with pytest.raises(ValueError, match="Model file does not exist"):
        build_controller(specs["player_1"])


def test_model_controller_decodes_first_legal_action() -> None:
    episode = run_multi_policy_episode(
        episode=0,
        player_names=("player_1", "player_2", "player_3"),
        controllers={
            "player_1": ModelController(
                FirstLegalActionModel(),
                policy_label="model:test.zip",
            ),
            "player_2": RandomMaskController(),
            "player_3": RandomMaskController(),
        },
        seed=1,
        max_turns=500,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
    )

    assert episode.turns > 0
    assert episode.winner in {"player_1", "player_2", "player_3"}


def test_evaluate_multi_policy_game_runs_three_random_players() -> None:
    result = evaluate_multi_policy_game(
        seat_policies={
            "player_1": "random",
            "player_2": "random",
            "player_3": "random",
        },
        episodes=3,
        players=3,
        seed=2,
        max_turns=500,
        wandb_mode="disabled",
    )

    assert result.episodes == 3
    assert len(result.seat_results) == 3
    assert sum(seat.wins for seat in result.seat_results) == 3
    assert result.random_combined_win_rate == 1.0


def test_summarize_evaluation_combines_trained_and_random_win_rates() -> None:
    controllers = {
        "player_1": ModelController(FirstLegalActionModel(), "model:a.zip"),
        "player_2": ModelController(FirstLegalActionModel(), "model:b.zip"),
        "player_3": RandomMaskController(),
    }
    episodes = tuple(
        run_multi_policy_episode(
            episode=index,
            player_names=("player_1", "player_2", "player_3"),
            controllers=controllers,
            seed=10 + index,
            max_turns=500,
            include_cards=None,
            exclude_cards=(),
            enabled_combo_rules=(),
            reveal_opponent_card_counts=False,
        )
        for index in range(3)
    )

    result = summarize_evaluation(
        episode_results=episodes,
        controllers=controllers,
        player_names=("player_1", "player_2", "player_3"),
    )

    assert result.trained_seats == ("player_1", "player_2")
    assert result.random_seats == ("player_3",)
    assert 0.0 <= result.trained_combined_win_rate <= 1.0
    assert 0.0 <= result.random_combined_win_rate <= 1.0


def test_format_results_table_includes_combined_rates() -> None:
    result = MultiPolicyEvaluationResult(
        episodes=1,
        players=3,
        seat_results=(
            SeatEvaluationResult(
                seat="player_1",
                policy_label="model:a.zip",
                wins=1,
                win_rate=1.0,
                average_reward=1.0,
                average_turns_survived=10.0,
                defuses=0,
                explosions=0,
                cards_played={},
            ),
        ),
        episode_results=(),
        trained_seats=("player_1",),
        trained_combined_wins=1,
        trained_combined_win_rate=1.0,
        random_seats=(),
        random_combined_wins=0,
        random_combined_win_rate=0.0,
    )

    table = format_results_table(result)

    assert "player_1" in table
    assert "trained_combined_win_rate=1.000" in table


def test_log_results_table_uses_wandb_table(monkeypatch) -> None:
    class FakeTable:
        def __init__(self, columns) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *row) -> None:
            self.rows.append(row)

    fake_wandb = SimpleNamespace(Table=FakeTable)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    result = MultiPolicyEvaluationResult(
        episodes=1,
        players=3,
        seat_results=(
            SeatEvaluationResult(
                seat="player_1",
                policy_label="random",
                wins=1,
                win_rate=1.0,
                average_reward=1.0,
                average_turns_survived=10.0,
                defuses=0,
                explosions=0,
                cards_played={},
            ),
        ),
        episode_results=(),
        trained_seats=(),
        trained_combined_wins=0,
        trained_combined_win_rate=0.0,
        random_seats=("player_1",),
        random_combined_wins=1,
        random_combined_win_rate=1.0,
    )
    run = FakeRun()

    log_results_table(run, result)

    table = run.logs[-1]["multi_policy/seat_comparison"]
    assert table.rows[0][0] == "player_1"
