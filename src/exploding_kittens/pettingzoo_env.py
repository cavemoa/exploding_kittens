"""PettingZoo AEC wrapper for the Exploding Kittens engine."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

from .action_encoding import ActionEncoder
from .engine import GameEngine, IllegalActionError
from .observation_encoding import ObservationEncoder


def env(**kwargs: Any) -> AECEnv:
    environment = raw_env(**kwargs)
    environment = wrappers.AssertOutOfBoundsWrapper(environment)
    environment = wrappers.OrderEnforcingWrapper(environment)
    return environment


class raw_env(AECEnv[str, dict[str, np.ndarray], int]):
    metadata = {
        "name": "exploding_kittens_v0",
        "render_modes": ["ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        *,
        players: int = 2,
        seed: int | None = None,
        max_turns: int = 500,
        include_cards: tuple[str, ...] | None = None,
        exclude_cards: tuple[str, ...] = (),
        enabled_combo_rules: tuple[str, ...] = (),
        reveal_opponent_card_counts: bool = False,
        max_hand_size: int = 32,
        render_mode: str | None = None,
    ) -> None:
        if players < 2:
            raise ValueError("players must be at least 2.")
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")
        if max_hand_size < 1:
            raise ValueError("max_hand_size must be at least 1.")
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'.")

        self.player_count = players
        self.initial_seed = seed
        self.max_turns = max_turns
        self.include_cards = include_cards
        self.exclude_cards = exclude_cards
        self.enabled_combo_rules = enabled_combo_rules
        self.reveal_opponent_card_counts = reveal_opponent_card_counts
        self.max_hand_size = max_hand_size
        self.render_mode = render_mode

        self.possible_agents = [
            f"player_{index + 1}" for index in range(self.player_count)
        ]
        player_names = tuple(self.possible_agents)
        self.action_encoder = ActionEncoder(
            max_hand_size=self.max_hand_size,
            max_players=self.player_count,
            player_names=player_names,
        )
        self.observation_encoder = ObservationEncoder(
            max_hand_size=self.max_hand_size,
            max_players=self.player_count,
            player_names=player_names,
            include_opponent_card_counts=self.reveal_opponent_card_counts,
            action_encoder=self.action_encoder,
        )
        self._action_spaces = {
            agent: gym.spaces.Discrete(self.action_encoder.action_space_size)
            for agent in self.possible_agents
        }
        self._observation_spaces = {
            agent: gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.observation_encoder.observation_size,),
                        dtype=np.float32,
                    ),
                    "action_mask": gym.spaces.MultiBinary(
                        self.action_encoder.action_space_size
                    ),
                }
            )
            for agent in self.possible_agents
        }
        self.engine: GameEngine | None = None
        self.agents: list[str] = []
        self.rewards: dict[str, float] = {}
        self._cumulative_rewards: dict[str, float] = {}
        self.terminations: dict[str, bool] = {}
        self.truncations: dict[str, bool] = {}
        self.infos: dict[str, dict[str, Any]] = {}
        self.episode_rewards: dict[str, float] = {}
        self._rewarded_eliminations: set[str] = set()
        self._winner_rewarded = False
        self.agent_selection: str | None = None

    @lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.Space:
        return self._observation_spaces[agent]

    @lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.Space:
        return self._action_spaces[agent]

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        del options
        game_seed = self.initial_seed if seed is None else seed
        self.engine = GameEngine.new_game(
            self.possible_agents,
            seed=game_seed,
            include_cards=self.include_cards,
            exclude_cards=self.exclude_cards,
            enabled_combo_rules=self.enabled_combo_rules,
            reveal_opponent_card_counts=self.reveal_opponent_card_counts,
        )
        self.agents = list(self.possible_agents)
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.episode_rewards = {agent: 0.0 for agent in self.agents}
        self._rewarded_eliminations = set()
        self._winner_rewarded = False
        self.infos = {agent: self._info_for(agent) for agent in self.agents}
        self.agent_selection = self.engine.current_player.name

    def observe(self, agent: str) -> dict[str, np.ndarray] | None:
        if self.engine is None or agent not in self.agents:
            return None
        return self.observation_encoder.encode(self.engine.observe(agent))

    def step(self, action: int | np.integer | None) -> None:
        if self.agent_selection is None:
            raise RuntimeError("Environment must be reset before stepping.")

        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        if action is None:
            raise ValueError("Live agents must provide an integer action.")
        if self.engine is None:
            raise RuntimeError("Environment must be reset before stepping.")

        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()

        observation = self.engine.observe(agent)
        game_action = self.action_encoder.decode(int(action), observation)
        try:
            events = self.engine.step(game_action)
        except IllegalActionError as exc:
            raise ValueError(str(exc)) from exc

        self._apply_sparse_rewards(events)
        self._update_done_flags()
        self._sync_infos()
        self._advance_agent_selection()
        self._accumulate_rewards()

    def render(self) -> str | None:
        if self.engine is None:
            return None

        winner = self.engine.winner or "none"
        lines = [
            f"Turn: {self.engine.turn_count}",
            f"Current player: {self.engine.current_player.name}",
            f"Winner: {winner}",
        ]
        rendered = "\n".join(lines)
        if self.render_mode == "ansi":
            return rendered
        return None

    def close(self) -> None:
        return None

    def _update_done_flags(self) -> None:
        if self.engine is None:
            return

        if self.engine.is_over:
            for agent in self.agents:
                self.terminations[agent] = True
            return

        if self.engine.turn_count >= self.max_turns:
            for agent in self.agents:
                self.truncations[agent] = True

    def _advance_agent_selection(self) -> None:
        if self.engine is None:
            return

        done_agents = [
            agent
            for agent in self.agents
            if self.terminations[agent] or self.truncations[agent]
        ]
        if done_agents:
            self.agent_selection = done_agents[0]
        else:
            self.agent_selection = self.engine.current_player.name

    def _apply_sparse_rewards(self, events: list[Any]) -> None:
        if self.engine is None:
            return

        for event in events:
            if event.kind == "eliminated" and event.player not in self._rewarded_eliminations:
                self._add_reward(event.player, -1.0)
                self._rewarded_eliminations.add(event.player)

        winner = self.engine.winner
        if self.engine.is_over and winner is not None and not self._winner_rewarded:
            self._add_reward(winner, 1.0)
            self._winner_rewarded = True

    def _add_reward(self, agent: str, amount: float) -> None:
        if agent not in self.rewards:
            return
        self.rewards[agent] += amount
        self.episode_rewards[agent] += amount

    def _sync_infos(self) -> None:
        self.infos = {agent: self._info_for(agent) for agent in self.agents}

    def _info_for(self, agent: str) -> dict[str, Any]:
        winner = self.engine.winner if self.engine is not None else None
        return {
            "episode_reward": self.episode_rewards.get(agent, 0.0),
            "winner": winner,
        }
