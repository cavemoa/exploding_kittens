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


class SafeRuleAgent:
    """A small rules-based baseline that uses only public observations."""

    danger_response_order = (
        Action.PLAY_SHUFFLE,
        Action.PLAY_SKIP,
        Action.PLAY_ATTACK,
        Action.PLAY_ALTER_THE_FUTURE,
    )
    low_value_combo_cards = ("normal",)

    def choose_action(
        self,
        observation: GameObservation,
        rng: random.Random,
    ) -> GameAction:
        if self._known_top_card_is_kitten(observation):
            return self._choose_known_kitten_response(observation)

        combo_action = self._choose_low_value_combo(observation)
        if combo_action is not None:
            return combo_action

        return self._draw_action(observation)

    def _choose_known_kitten_response(self, observation: GameObservation) -> GameAction:
        for action_kind in self.danger_response_order:
            action = self._first_action_of_kind(observation, action_kind)
            if action is not None:
                return action

        non_defuse_action = self._first_non_draw_action_without_defuse(observation)
        if non_defuse_action is not None:
            return non_defuse_action

        return self._draw_action(observation)

    def _choose_low_value_combo(self, observation: GameObservation) -> GameAction | None:
        for action, card_names in observation.legal_actions_with_card_names():
            if action.kind is not Action.PLAY_TWO_OF_A_KIND:
                continue
            if self._all_low_value_cards(card_names):
                return action
        return None

    def _first_non_draw_action_without_defuse(
        self,
        observation: GameObservation,
    ) -> GameAction | None:
        for action, card_names in observation.legal_actions_with_card_names():
            if action.kind is Action.DRAW:
                continue
            if "defuse" not in card_names:
                return action
        return None

    def _first_action_of_kind(
        self,
        observation: GameObservation,
        action_kind: Action,
    ) -> GameAction | None:
        return next(
            (action for action in observation.legal_actions if action.kind is action_kind),
            None,
        )

    def _draw_action(self, observation: GameObservation) -> GameAction:
        return self._first_action_of_kind(observation, Action.DRAW) or GameAction(
            kind=Action.DRAW
        )

    def _known_top_card_is_kitten(self, observation: GameObservation) -> bool:
        return bool(observation.known_top_cards) and (
            observation.known_top_cards[0] == "exploding_kitten"
        )

    def _all_low_value_cards(self, card_names: tuple[str, ...]) -> bool:
        return bool(card_names) and all(
            card_name in self.low_value_combo_cards for card_name in card_names
        )
