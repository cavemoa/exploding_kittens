from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_multi_policy_game import RandomMaskController, run_multi_policy_episode
from scripts.evaluate_rllib_checkpoint import (
    RLLibController,
    RLLibCheckpointEvaluationResult,
    build_controllers,
    format_results_table,
    log_results_table,
    summarize_rllib_evaluation,
    validate_evaluation_args,
)
from scripts.evaluate_multi_policy_game import (
    SeatEvaluationResult,
    summarize_evaluation,
)


class FirstLegalRLLibPolicy:
    def compute_single_action(self, observation, *, explore=False):
        del explore
        action_id = int(np.flatnonzero(observation["action_mask"])[0])
        return action_id, [], {}


class IllegalRLLibPolicy:
    def compute_single_action(self, observation, *, explore=False):
        del observation, explore
        return 999, [], {}


class FakeRun:
    def __init__(self) -> None:
        self.logs = []

    def log(self, data) -> None:
        self.logs.append(data)


def test_rllib_controller_decodes_first_legal_action() -> None:
    episode = run_multi_policy_episode(
        episode=0,
        player_names=("player_1", "player_2", "player_3"),
        controllers={
            "player_1": RLLibController(
                policy=FirstLegalRLLibPolicy(),
                policy_label="rllib:test",
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


def test_rllib_controller_rejects_illegal_policy_action() -> None:
    with pytest.raises(ValueError, match="illegal action"):
        run_multi_policy_episode(
            episode=0,
            player_names=("player_1", "player_2", "player_3"),
            controllers={
                "player_1": RLLibController(
                    policy=IllegalRLLibPolicy(),
                    policy_label="rllib:test",
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


def test_build_controllers_accepts_scripted_and_rejects_missing_rllib_checkpoint() -> None:
    controllers = build_controllers(
        {
            "player_1": "safe-rule",
            "player_2": "random",
        }
    )

    assert controllers["player_1"].policy_label == "safe-rule"
    assert controllers["player_2"].policy_label == "random"

    with pytest.raises(ValueError, match="checkpoint does not exist"):
        build_controllers({"player_1": "rllib:models/missing_checkpoint"})


def test_validate_evaluation_args_requires_rllib_policy() -> None:
    with pytest.raises(ValueError, match="rllib"):
        validate_evaluation_args(
            seat_policies={"player_1": "random"},
            episodes=1,
            players=3,
            wandb_mode="disabled",
        )


def test_summarize_rllib_evaluation_tracks_grouped_rates() -> None:
    controllers = {
        "player_1": RLLibController(
            policy=FirstLegalRLLibPolicy(),
            policy_label="rllib:a",
        ),
        "player_2": RandomMaskController(),
        "player_3": RandomMaskController(),
    }
    episodes = tuple(
        run_multi_policy_episode(
            episode=index,
            player_names=("player_1", "player_2", "player_3"),
            controllers=controllers,
            seed=3 + index,
            max_turns=500,
            include_cards=None,
            exclude_cards=(),
            enabled_combo_rules=(),
            reveal_opponent_card_counts=False,
        )
        for index in range(3)
    )
    base_result = summarize_evaluation(
        episode_results=episodes,
        controllers=controllers,
        player_names=("player_1", "player_2", "player_3"),
    )

    result = summarize_rllib_evaluation(
        base_result=base_result,
        controllers=controllers,
        player_names=("player_1", "player_2", "player_3"),
    )

    assert result.rllib_seats == ("player_1",)
    assert result.scripted_opponent_seats == ("player_2", "player_3")
    assert result.rllib_combined_wins + result.scripted_opponent_wins == 3
    assert 0.0 <= result.seat_order_advantage <= 1.0


def test_format_results_table_includes_rllib_grouped_rates() -> None:
    base_result = SimpleNamespace(
        episodes=1,
        players=3,
        seat_results=(
            SeatEvaluationResult(
                seat="player_1",
                policy_label="rllib:a",
                wins=1,
                win_rate=1.0,
                average_reward=1.0,
                average_turns_survived=5.0,
                defuses=0,
                explosions=0,
                cards_played={},
            ),
        ),
        episode_results=(),
        trained_combined_win_rate=1.0,
        random_combined_win_rate=0.0,
    )
    result = RLLibCheckpointEvaluationResult(
        base_result=base_result,
        rllib_seats=("player_1",),
        rllib_combined_wins=1,
        rllib_combined_win_rate=1.0,
        scripted_opponent_seats=(),
        scripted_opponent_wins=0,
        scripted_opponent_win_rate=0.0,
        seat_order_advantage=0.0,
    )

    table = format_results_table(result)

    assert "rllib_combined_win_rate=1.000" in table
    assert "scripted_opponent_win_rate=0.000" in table


def test_log_results_table_uses_wandb_table(monkeypatch) -> None:
    class FakeTable:
        def __init__(self, columns) -> None:
            self.columns = columns
            self.rows = []

        def add_data(self, *row) -> None:
            self.rows.append(row)

    fake_wandb = SimpleNamespace(Table=FakeTable)
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    base_result = SimpleNamespace(
        seat_results=(
            SeatEvaluationResult(
                seat="player_1",
                policy_label="rllib:a",
                wins=1,
                win_rate=1.0,
                average_reward=1.0,
                average_turns_survived=5.0,
                defuses=0,
                explosions=0,
                cards_played={},
            ),
        )
    )
    result = RLLibCheckpointEvaluationResult(
        base_result=base_result,
        rllib_seats=("player_1",),
        rllib_combined_wins=1,
        rllib_combined_win_rate=1.0,
        scripted_opponent_seats=(),
        scripted_opponent_wins=0,
        scripted_opponent_win_rate=0.0,
        seat_order_advantage=0.0,
    )
    run = FakeRun()

    log_results_table(run, result)

    table = run.logs[-1]["rllib_eval/seat_comparison"]
    assert table.rows[0][0] == "player_1"
