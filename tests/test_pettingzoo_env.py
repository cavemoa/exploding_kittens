import warnings

import pytest
from pettingzoo.test import api_test

from exploding_kittens import ExplodingKittenCard, NormalCard, RawPettingZooEnv, pettingzoo_env


def test_pettingzoo_env_passes_api_test() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Observation is not a NumPy array")
        warnings.filterwarnings(
            "ignore",
            message="Observation space for each agent probably should be",
        )
        api_test(pettingzoo_env(players=2, max_turns=50), num_cycles=100)


def test_pettingzoo_observation_contains_numeric_vector_and_action_mask() -> None:
    environment = RawPettingZooEnv(players=2, max_turns=50)
    environment.reset(seed=1)

    observation, reward, termination, truncation, info = environment.last()

    assert reward == 0
    assert not termination
    assert not truncation
    assert info == {"episode_reward": 0.0, "winner": None}
    assert set(observation) == {"observation", "action_mask"}
    assert environment.observation_space(environment.agent_selection).contains(observation)
    assert observation["action_mask"][0] == 1


def test_pettingzoo_step_decodes_masked_action_and_advances_turn() -> None:
    environment = RawPettingZooEnv(players=2, max_turns=50)
    environment.reset(seed=1)
    first_agent = environment.agent_selection
    observation = environment.last()[0]
    draw_action_id = 0

    assert observation["action_mask"][draw_action_id] == 1
    environment.step(draw_action_id)

    assert environment.agent_selection != first_agent
    assert environment.engine.turn_count == 1


def test_pettingzoo_truncates_all_agents_at_max_turns() -> None:
    environment = RawPettingZooEnv(players=2, max_turns=1)
    environment.reset(seed=1)

    environment.step(0)

    assert all(environment.truncations.values())
    assert not any(environment.terminations.values())
    assert environment.rewards == {"player_1": 0.0, "player_2": 0.0}
    assert environment.episode_rewards == {"player_1": 0.0, "player_2": 0.0}


def test_pettingzoo_rewards_eliminated_player_before_game_is_over() -> None:
    environment = RawPettingZooEnv(players=3, max_turns=50)
    environment.reset(seed=1)
    current_player = environment.engine.current_player
    current_player.hand.clear()
    environment.engine.draw_pile = [NormalCard(), ExplodingKittenCard()]

    environment.step(0)

    assert environment.rewards[current_player.name] == -1.0
    assert environment.episode_rewards[current_player.name] == -1.0
    assert not any(environment.terminations.values())
    assert environment.infos[current_player.name]["episode_reward"] == -1.0


def test_pettingzoo_terminal_rank_profile_delays_elimination_rewards() -> None:
    environment = RawPettingZooEnv(
        players=3,
        max_turns=50,
        reward_profile="terminal-rank",
    )
    environment.reset(seed=1)
    current_player = environment.engine.current_player
    current_player.hand.clear()
    environment.engine.draw_pile = [NormalCard(), ExplodingKittenCard()]

    environment.step(0)

    assert environment.rewards[current_player.name] == 0.0
    assert environment.episode_rewards[current_player.name] == 0.0
    assert environment._elimination_order == [current_player.name]
    assert not any(environment.terminations.values())


def test_pettingzoo_rewards_winner_and_eliminated_player_when_game_is_over() -> None:
    environment = RawPettingZooEnv(players=2, max_turns=50)
    environment.reset(seed=1)
    current_player = environment.engine.current_player
    current_player.hand.clear()
    environment.engine.draw_pile = [ExplodingKittenCard()]

    environment.step(0)

    winner = environment.engine.winner
    loser = current_player.name
    assert all(environment.terminations.values())
    assert not any(environment.truncations.values())
    assert winner is not None
    assert environment.rewards[winner] == 1.0
    assert environment.rewards[loser] == -1.0
    assert environment.episode_rewards[winner] == 1.0
    assert environment.episode_rewards[loser] == -1.0
    assert environment.infos[winner] == {"episode_reward": 1.0, "winner": winner}
    assert environment.infos[loser] == {"episode_reward": -1.0, "winner": winner}


def test_pettingzoo_terminal_rank_rewards_follow_elimination_order() -> None:
    environment = RawPettingZooEnv(
        players=4,
        max_turns=50,
        reward_profile="terminal-rank",
        terminal_rank_rewards=(1.0, 0.3, -0.3, -1.0),
    )
    environment.reset(seed=1)
    environment._elimination_order = ["player_3", "player_1", "player_4"]
    for player in environment.engine.players:
        player.alive = player.name == "player_2"

    environment._apply_terminal_rank_rewards()

    assert environment._ranked_agents() == (
        "player_2",
        "player_4",
        "player_1",
        "player_3",
    )
    assert environment.rewards == {
        "player_1": -0.3,
        "player_2": 1.0,
        "player_3": -1.0,
        "player_4": 0.3,
    }
    assert environment.episode_rewards == environment.rewards


def test_pettingzoo_terminal_rank_rewards_validate_weight_count() -> None:
    with pytest.raises(ValueError, match="terminal_rank_rewards"):
        RawPettingZooEnv(
            players=4,
            reward_profile="terminal-rank",
            terminal_rank_rewards=(1.0, -1.0),
        )
