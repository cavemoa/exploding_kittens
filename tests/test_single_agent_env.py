import numpy as np
import pytest

from exploding_kittens import ExplodingKittenCard, NormalCard, SingleAgentEnv


def legal_action_ids(action_mask: np.ndarray) -> list[int]:
    return np.flatnonzero(action_mask).astype(int).tolist()


def test_single_agent_env_reset_returns_learner_observation_and_mask() -> None:
    environment = SingleAgentEnv(players=3, opponent_strategy="draw-only")

    observation, info = environment.reset(seed=1)

    assert set(observation) == {"observation", "action_mask"}
    assert environment.observation_space.contains(observation)
    assert observation["action_mask"][0] == 1
    assert info["episode_reward"] == 0.0
    assert info["learner"] == "player_1"


def test_single_agent_env_applies_learner_action() -> None:
    environment = SingleAgentEnv(players=2, opponent_strategy="draw-only")
    observation, _ = environment.reset(seed=1)
    starting_turn_count = environment.engine.turn_count

    assert observation["action_mask"][0] == 1
    _, reward, terminated, truncated, info = environment.step(0)

    assert environment.engine.turn_count > starting_turn_count
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["turn_count"] == environment.engine.turn_count


def test_single_agent_env_auto_plays_opponents_until_learner_turn() -> None:
    environment = SingleAgentEnv(players=3, opponent_strategy="draw-only")
    environment.reset(seed=2)

    _, _, terminated, truncated, _ = environment.step(0)

    assert environment.engine.turn_count >= 3
    if not terminated and not truncated:
        assert environment.engine.current_player.name == "player_1"


def test_single_agent_env_terminal_reward_when_learner_explodes() -> None:
    environment = SingleAgentEnv(players=2, opponent_strategy="draw-only")
    environment.reset(seed=1)
    learner = environment.engine.player_by_name("player_1")
    learner.hand.clear()
    environment.engine.draw_pile = [ExplodingKittenCard()]

    _, reward, terminated, truncated, info = environment.step(0)

    assert reward == -1.0
    assert terminated
    assert not truncated
    assert info["episode_reward"] == -1.0
    assert info["winner"] == "player_2"


def test_single_agent_env_terminal_reward_when_learner_wins_during_opponent_turn() -> None:
    environment = SingleAgentEnv(players=2, opponent_strategy="draw-only")
    environment.reset(seed=1)
    opponent = environment.engine.player_by_name("player_2")
    opponent.hand.clear()
    environment.engine.draw_pile = [ExplodingKittenCard(), NormalCard()]

    _, reward, terminated, truncated, info = environment.step(0)

    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["episode_reward"] == 1.0
    assert info["winner"] == "player_1"


def test_single_agent_env_rejects_illegal_masked_action_id() -> None:
    environment = SingleAgentEnv(players=2, opponent_strategy="draw-only")
    observation, _ = environment.reset(seed=1)
    illegal_ids = [
        action_id
        for action_id in range(environment.action_space.n)
        if action_id not in legal_action_ids(observation["action_mask"])
    ]

    with pytest.raises(ValueError, match="not legal"):
        environment.step(illegal_ids[0])


def test_single_agent_env_supports_scripted_opponent_strategies() -> None:
    for strategy in ("random", "draw-only", "safe-rule", "skip-if-possible"):
        environment = SingleAgentEnv(players=3, opponent_strategy=strategy)
        observation, info = environment.reset(seed=3)

        assert environment.observation_space.contains(observation)
        assert info["opponent_strategy"] == strategy
