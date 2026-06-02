"""Single-agent Gymnasium wrapper with scripted opponents."""

from __future__ import annotations

from typing import Any
import random

import gymnasium as gym
import numpy as np

from .action_encoding import ActionEncoder
from .agents import Agent, DrawOnlyAgent, RandomAgent, SafeRuleAgent, SkipIfPossibleAgent
from .engine import GameEngine, GameEvent, IllegalActionError
from .observation_encoding import ObservationEncoder


SCRIPTED_OPPONENTS: dict[str, type[Agent]] = {
    "draw-only": DrawOnlyAgent,
    "random": RandomAgent,
    "safe-rule": SafeRuleAgent,
    "skip-if-possible": SkipIfPossibleAgent,
}


class SingleAgentEnv(gym.Env[dict[str, np.ndarray], int]):
    """Gymnasium environment for training one learner against fixed opponents."""

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        *,
        players: int = 3,
        learner: str = "player_1",
        opponent_strategy: str = "safe-rule",
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
        if opponent_strategy not in SCRIPTED_OPPONENTS:
            known = ", ".join(sorted(SCRIPTED_OPPONENTS))
            raise ValueError(
                f"Unknown opponent_strategy: {opponent_strategy}. Known strategies: {known}."
            )
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'.")

        self.player_count = players
        self.learner = learner
        self.opponent_strategy = opponent_strategy
        self.initial_seed = seed
        self.max_turns = max_turns
        self.include_cards = include_cards
        self.exclude_cards = exclude_cards
        self.enabled_combo_rules = enabled_combo_rules
        self.reveal_opponent_card_counts = reveal_opponent_card_counts
        self.max_hand_size = max_hand_size
        self.render_mode = render_mode

        self.player_names = tuple(
            f"player_{index + 1}" for index in range(self.player_count)
        )
        if self.learner not in self.player_names:
            raise ValueError(f"learner must be one of: {', '.join(self.player_names)}.")

        self.action_encoder = ActionEncoder(
            max_hand_size=self.max_hand_size,
            max_players=self.player_count,
            player_names=self.player_names,
        )
        self.observation_encoder = ObservationEncoder(
            max_hand_size=self.max_hand_size,
            max_players=self.player_count,
            player_names=self.player_names,
            include_opponent_card_counts=self.reveal_opponent_card_counts,
            action_encoder=self.action_encoder,
        )
        self.action_space = gym.spaces.Discrete(self.action_encoder.action_space_size)
        self.observation_space = gym.spaces.Dict(
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

        self.engine: GameEngine | None = None
        self.opponents: dict[str, Agent] = {}
        self.opponent_rng = random.Random()
        self.episode_reward = 0.0
        self.terminated = False
        self.truncated = False
        self._winner_rewarded = False
        self._learner_elimination_rewarded = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        del options
        super().reset(seed=seed)
        game_seed = self._game_seed(seed)
        self.engine = GameEngine.new_game(
            self.player_names,
            seed=game_seed,
            include_cards=self.include_cards,
            exclude_cards=self.exclude_cards,
            enabled_combo_rules=self.enabled_combo_rules,
            reveal_opponent_card_counts=self.reveal_opponent_card_counts,
        )
        opponent_type = SCRIPTED_OPPONENTS[self.opponent_strategy]
        self.opponents = {
            player_name: opponent_type()
            for player_name in self.player_names
            if player_name != self.learner
        }
        self.opponent_rng = random.Random(game_seed)
        self.episode_reward = 0.0
        self.terminated = False
        self.truncated = False
        self._winner_rewarded = False
        self._learner_elimination_rewarded = False

        auto_reward = self._play_scripted_until_learner_turn()
        self.episode_reward += auto_reward
        return self._encoded_observation(), self._info()

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self.terminated or self.truncated:
            raise RuntimeError("Cannot step a terminated or truncated episode.")
        if self.engine.current_player.name != self.learner:
            raise RuntimeError("Learner cannot act when it is not the learner's turn.")

        observation = self.engine.observe(self.learner)
        game_action = self.action_encoder.decode(int(action), observation)
        try:
            events = self.engine.step(game_action)
        except IllegalActionError as exc:
            raise ValueError(str(exc)) from exc

        reward = self._reward_from_events(events)
        self._update_done_flags()
        if not self.terminated and not self.truncated:
            reward += self._play_scripted_until_learner_turn()

        self.episode_reward += reward
        return (
            self._encoded_observation(),
            reward,
            self.terminated,
            self.truncated,
            self._info(),
        )

    def render(self) -> str | None:
        if self.engine is None:
            return None

        rendered = "\n".join(
            [
                f"Turn: {self.engine.turn_count}",
                f"Learner: {self.learner}",
                f"Current player: {self.engine.current_player.name}",
                f"Winner: {self.engine.winner or 'none'}",
            ]
        )
        if self.render_mode == "ansi":
            return rendered
        return None

    def close(self) -> None:
        return None

    def action_masks(self) -> np.ndarray:
        """Return legal action masks for sb3-contrib MaskablePPO."""

        return self._encoded_observation()["action_mask"].astype(bool)

    def _game_seed(self, seed: int | None) -> int:
        if seed is not None:
            return seed
        if self.initial_seed is not None:
            return self.initial_seed
        return int(self.np_random.integers(0, 2**32 - 1))

    def _play_scripted_until_learner_turn(self) -> float:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before stepping.")

        reward = 0.0
        while (
            not self.terminated
            and not self.truncated
            and not self.engine.is_over
            and self.engine.current_player.name != self.learner
        ):
            player_name = self.engine.current_player.name
            opponent = self.opponents[player_name]
            observation = self.engine.observe(player_name)
            action = opponent.choose_action(observation, self.opponent_rng)
            events = self.engine.step(action)
            reward += self._reward_from_events(events)
            self._update_done_flags()
        return reward

    def _reward_from_events(self, events: list[GameEvent]) -> float:
        if self.engine is None:
            return 0.0

        reward = 0.0
        for event in events:
            if (
                event.kind == "eliminated"
                and event.player == self.learner
                and not self._learner_elimination_rewarded
            ):
                reward -= 1.0
                self._learner_elimination_rewarded = True

        if (
            self.engine.is_over
            and self.engine.winner == self.learner
            and not self._winner_rewarded
        ):
            reward += 1.0
            self._winner_rewarded = True
        return reward

    def _update_done_flags(self) -> None:
        if self.engine is None:
            return

        learner_state = self.engine.player_by_name(self.learner)
        self.terminated = self.engine.is_over or learner_state is None or not learner_state.alive
        self.truncated = not self.terminated and self.engine.turn_count >= self.max_turns

    def _encoded_observation(self) -> dict[str, np.ndarray]:
        if self.engine is None:
            raise RuntimeError("Environment must be reset before observing.")
        return self.observation_encoder.encode(self.engine.observe(self.learner))

    def _info(self) -> dict[str, Any]:
        winner = self.engine.winner if self.engine is not None else None
        return {
            "episode_reward": self.episode_reward,
            "winner": winner,
            "turn_count": self.engine.turn_count if self.engine is not None else 0,
            "learner": self.learner,
            "opponent_strategy": self.opponent_strategy,
        }
