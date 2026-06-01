"""Simple baseline agents for exercising the engine."""

from __future__ import annotations

import random
from typing import Protocol

from .actions import Action, GameAction
from .observations import GameObservation


class Agent(Protocol):
    def choose_action(
        self,
        observation: GameObservation,
        rng: random.Random,
    ) -> GameAction:
        """Choose one legal action for the current player."""


class RandomAgent:
    def choose_action(
        self,
        observation: GameObservation,
        rng: random.Random,
    ) -> GameAction:
        return rng.choice(observation.legal_actions)


class DrawOnlyAgent:
    def choose_action(
        self,
        observation: GameObservation,
        rng: random.Random,
    ) -> GameAction:
        return GameAction(kind=Action.DRAW)


class SkipIfPossibleAgent:
    def choose_action(
        self,
        observation: GameObservation,
        rng: random.Random,
    ) -> GameAction:
        for action in observation.legal_actions:
            if action.kind is Action.PLAY_SKIP:
                return action
        return GameAction(kind=Action.DRAW)
