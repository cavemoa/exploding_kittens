"""Numeric encoding for agent observations."""

from __future__ import annotations

import numpy as np

from .action_encoding import ActionEncoder
from .cards import CARD_TYPES
from .observations import GameObservation


class ObservationEncoder:
    """Convert `GameObservation` objects into fixed-shape numeric arrays."""

    def __init__(
        self,
        *,
        max_hand_size: int = 32,
        max_players: int = 5,
        player_names: tuple[str, ...] | None = None,
        max_draw_pile_size: int = 128,
        max_turn_debt: int = 8,
        known_top_card_count: int = 3,
        include_opponent_card_counts: bool = False,
        action_encoder: ActionEncoder | None = None,
    ) -> None:
        if max_hand_size < 1:
            raise ValueError("max_hand_size must be at least 1.")
        if max_players < 2:
            raise ValueError("max_players must be at least 2.")
        if max_draw_pile_size < 1:
            raise ValueError("max_draw_pile_size must be at least 1.")
        if max_turn_debt < 1:
            raise ValueError("max_turn_debt must be at least 1.")
        if known_top_card_count < 0:
            raise ValueError("known_top_card_count cannot be negative.")

        self.max_hand_size = max_hand_size
        self.max_players = max_players
        self.player_names = player_names or tuple(
            f"player_{index + 1}" for index in range(max_players)
        )
        if len(self.player_names) > max_players:
            raise ValueError("player_names cannot contain more than max_players entries.")

        self.max_draw_pile_size = max_draw_pile_size
        self.max_turn_debt = max_turn_debt
        self.known_top_card_count = known_top_card_count
        self.include_opponent_card_counts = include_opponent_card_counts
        self.card_names = tuple(card_type.name for card_type in CARD_TYPES)
        self.card_vocab_size = len(self.card_names) + 1
        self.card_index_by_name = {
            card_name: card_index + 1
            for card_index, card_name in enumerate(self.card_names)
        }
        self.action_encoder = action_encoder or ActionEncoder(
            max_hand_size=max_hand_size,
            max_players=max_players,
            player_names=self.player_names,
        )

        self.own_hand_size = self.max_hand_size * self.card_vocab_size
        self.hand_counts_size = len(self.card_names)
        self.discard_counts_size = len(self.card_names)
        self.scalar_size = 4
        self.known_top_cards_size = self.known_top_card_count * self.card_vocab_size
        self.opponent_counts_size = (
            max_players - 1 if self.include_opponent_card_counts else 0
        )
        self.observation_size = (
            self.own_hand_size
            + self.hand_counts_size
            + self.discard_counts_size
            + self.scalar_size
            + self.known_top_cards_size
            + self.opponent_counts_size
        )

    def encode(self, observation: GameObservation) -> dict[str, np.ndarray]:
        return {
            "observation": self.encode_observation(observation),
            "action_mask": self.action_encoder.action_mask(observation),
        }

    def encode_observation(self, observation: GameObservation) -> np.ndarray:
        parts = [
            self._encode_card_slots(observation.own_hand, self.max_hand_size),
            self._encode_counts(observation.hand_counts, self.max_hand_size),
            self._encode_counts(observation.discard_counts, self.max_draw_pile_size),
            self._encode_scalars(observation),
            self._encode_card_slots(
                observation.known_top_cards,
                self.known_top_card_count,
            ),
        ]
        if self.include_opponent_card_counts:
            parts.append(self._encode_opponent_card_counts(observation))

        encoded = np.concatenate(parts).astype(np.float32)
        if encoded.shape != (self.observation_size,):
            raise ValueError("Encoded observation did not match configured shape.")
        return encoded

    def _encode_card_slots(
        self,
        card_names: tuple[str, ...],
        slot_count: int,
    ) -> np.ndarray:
        if len(card_names) > slot_count:
            raise ValueError(f"Cannot encode {len(card_names)} cards in {slot_count} slots.")

        encoded = np.zeros(slot_count * self.card_vocab_size, dtype=np.float32)
        for slot_index, card_name in enumerate(card_names):
            card_index = self._card_index(card_name)
            encoded[slot_index * self.card_vocab_size + card_index] = 1.0
        for slot_index in range(len(card_names), slot_count):
            encoded[slot_index * self.card_vocab_size] = 1.0
        return encoded

    def _encode_counts(
        self,
        counts: dict[str, int],
        normalizer: int,
    ) -> np.ndarray:
        return np.array(
            [
                self._clamp(counts.get(card_name, 0) / normalizer)
                for card_name in self.card_names
            ],
            dtype=np.float32,
        )

    def _encode_scalars(self, observation: GameObservation) -> np.ndarray:
        return np.array(
            [
                self._clamp(observation.draw_pile_size / self.max_draw_pile_size),
                self._clamp(observation.living_players / self.max_players),
                self._clamp(observation.current_turns_remaining / self.max_turn_debt),
                self._clamp(observation.next_player_turns / self.max_turn_debt),
            ],
            dtype=np.float32,
        )

    def _encode_opponent_card_counts(self, observation: GameObservation) -> np.ndarray:
        opponents = [
            player_name
            for player_name in self.player_names
            if player_name != observation.player
        ][: self.max_players - 1]
        while len(opponents) < self.max_players - 1:
            opponents.append("")
        return np.array(
            [
                self._clamp(
                    observation.opponent_card_counts.get(player_name, 0)
                    / self.max_hand_size
                )
                for player_name in opponents
            ],
            dtype=np.float32,
        )

    def _card_index(self, card_name: str) -> int:
        try:
            return self.card_index_by_name[card_name]
        except KeyError as exc:
            raise ValueError(f"Unknown card name: {card_name}.") from exc

    def _clamp(self, value: float) -> float:
        return max(0.0, min(1.0, value))
