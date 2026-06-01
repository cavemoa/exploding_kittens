"""Stable integer encoding for concrete game actions."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .actions import Action, GameAction
from .observations import GameObservation


class ActionEncoder:
    """Map legal `GameAction` objects to stable integer action IDs.

    Layout:
    - 0: draw
    - hand slot: play one untargeted card from that slot
    - hand slot + target: play one targeted card from that slot
    - two hand slots + target: play a two-card combo
    - reserved tail: future named-card choices
    """

    def __init__(
        self,
        *,
        max_hand_size: int = 32,
        max_players: int = 5,
        player_names: tuple[str, ...] | None = None,
        reserved_named_card_actions: int = 64,
    ) -> None:
        if max_hand_size < 1:
            raise ValueError("max_hand_size must be at least 1.")
        if max_players < 2:
            raise ValueError("max_players must be at least 2.")
        if reserved_named_card_actions < 0:
            raise ValueError("reserved_named_card_actions cannot be negative.")

        self.max_hand_size = max_hand_size
        self.max_players = max_players
        self.player_names = player_names or tuple(
            f"player_{index + 1}" for index in range(max_players)
        )
        if len(self.player_names) > max_players:
            raise ValueError("player_names cannot contain more than max_players entries.")

        self.reserved_named_card_actions = reserved_named_card_actions
        self.draw_action_id = 0
        self.single_card_start = 1
        self.targeted_card_start = self.single_card_start + self.max_hand_size
        self.combo_start = (
            self.targeted_card_start + self.max_hand_size * self.max_players
        )
        self.combo_pair_count = self.max_hand_size * (self.max_hand_size - 1) // 2
        self.reserved_named_card_start = (
            self.combo_start + self.combo_pair_count * self.max_players
        )
        self.action_space_size = (
            self.reserved_named_card_start + self.reserved_named_card_actions
        )

    def encode(self, action: GameAction, observation: GameObservation) -> int:
        if action not in observation.legal_actions:
            raise ValueError(f"{action} is not legal for {observation.player}.")
        if action.named_card is not None:
            raise ValueError("Named-card actions are reserved but not implemented yet.")
        if action.kind is Action.DRAW:
            return self.draw_action_id

        if len(action.card_indexes) == 1 and action.target_player is None:
            card_index = action.card_indexes[0]
            self._require_card_index(card_index)
            return self.single_card_start + card_index

        if len(action.card_indexes) == 1 and action.target_player is not None:
            card_index = action.card_indexes[0]
            self._require_card_index(card_index)
            target_index = self._target_index(action.target_player)
            return self.targeted_card_start + card_index * self.max_players + target_index

        if len(action.card_indexes) == 2 and action.target_player is not None:
            first_index, second_index = sorted(action.card_indexes)
            self._require_card_index(first_index)
            self._require_card_index(second_index)
            pair_index = self._pair_to_index(first_index, second_index)
            target_index = self._target_index(action.target_player)
            return self.combo_start + pair_index * self.max_players + target_index

        raise ValueError(f"{action} cannot be represented by the current action encoder.")

    def decode(self, action_id: int, observation: GameObservation) -> GameAction:
        self._require_action_id(action_id)
        for action in observation.legal_actions:
            if self.encode(action, observation) == action_id:
                return action
        raise ValueError(f"Action ID {action_id} is not legal for {observation.player}.")

    def action_mask(self, observation: GameObservation) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        for action in observation.legal_actions:
            mask[self.encode(action, observation)] = 1
        return mask

    def _require_action_id(self, action_id: int) -> None:
        if not 0 <= action_id < self.action_space_size:
            raise ValueError(f"Action ID {action_id} is outside the action space.")

    def _require_card_index(self, card_index: int) -> None:
        if not 0 <= card_index < self.max_hand_size:
            raise ValueError(
                f"Card index {card_index} is outside max_hand_size={self.max_hand_size}."
            )

    def _target_index(self, target_player: str) -> int:
        try:
            target_index = self.player_names.index(target_player)
        except ValueError as exc:
            raise ValueError(f"Unknown target player: {target_player}.") from exc
        if not 0 <= target_index < self.max_players:
            raise ValueError(f"Target player {target_player} is outside max_players.")
        return target_index

    def _pair_to_index(self, first_index: int, second_index: int) -> int:
        if first_index >= second_index:
            raise ValueError("Combo card indexes must contain two distinct hand slots.")

        for pair_index, pair in enumerate(combinations(range(self.max_hand_size), 2)):
            if pair == (first_index, second_index):
                return pair_index
        raise ValueError(f"Card pair {(first_index, second_index)} is out of range.")
