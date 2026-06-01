import numpy as np
import pytest

from scripts.run_pettingzoo_random import run_random_rollout, sample_masked_action


def test_random_pettingzoo_rollout_finishes_game() -> None:
    result = run_random_rollout(
        players=2,
        seed=1,
        max_turns=500,
        enabled_combo_rules=("two_of_a_kind",),
    )

    assert result.completed
    assert not result.truncated
    assert result.winner in {"player_1", "player_2"}
    assert result.turns > 0


def test_random_pettingzoo_rollout_samples_no_illegal_actions() -> None:
    result = run_random_rollout(
        players=3,
        seed=2,
        max_turns=500,
        enabled_combo_rules=("two_of_a_kind",),
    )

    assert result.illegal_actions_sampled == 0


def test_random_pettingzoo_rollout_reward_totals_are_reasonable() -> None:
    result = run_random_rollout(players=3, seed=3, max_turns=500)

    assert set(result.episode_rewards) == {"player_1", "player_2", "player_3"}
    assert sum(1 for reward in result.episode_rewards.values() if reward == 1.0) == 1
    assert sum(1 for reward in result.episode_rewards.values() if reward == -1.0) == 2
    assert sum(result.episode_rewards.values()) == -1.0


def test_random_pettingzoo_rollout_is_deterministic_for_fixed_seed() -> None:
    first = run_random_rollout(
        players=3,
        seed=4,
        max_turns=500,
        enabled_combo_rules=("two_of_a_kind",),
    )
    second = run_random_rollout(
        players=3,
        seed=4,
        max_turns=500,
        enabled_combo_rules=("two_of_a_kind",),
    )

    assert first == second


def test_sample_masked_action_only_returns_legal_ids() -> None:
    rng = np.random.default_rng(1)
    action_mask = np.array([0, 1, 0, 1, 0], dtype=np.int8)

    samples = {sample_masked_action(action_mask, rng) for _ in range(50)}

    assert samples <= {1, 3}


def test_sample_masked_action_rejects_empty_masks() -> None:
    with pytest.raises(ValueError, match="empty action mask"):
        sample_masked_action(np.zeros(5, dtype=np.int8), np.random.default_rng(1))
