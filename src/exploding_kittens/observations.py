"""Observation objects exposed to agents."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import GameAction


@dataclass(frozen=True)
class GameObservation:
    player: str
    own_hand: tuple[str, ...]
    hand_counts: dict[str, int]
    discard_counts: dict[str, int]
    draw_pile_size: int
    living_players: int
    current_turns_remaining: int
    next_player_turns: int
    known_top_cards: tuple[str, ...]
    opponent_card_counts: dict[str, int]
    public_events: tuple[str, ...]
    legal_actions: tuple[GameAction, ...]

    def card_names_for_action(self, action: GameAction) -> tuple[str, ...]:
        """Return the current player's own card names referenced by an action."""
        try:
            return tuple(self.own_hand[card_index] for card_index in action.card_indexes)
        except IndexError as exc:
            raise ValueError(
                f"{action} references a card outside {self.player}'s hand."
            ) from exc

    def legal_actions_with_card_names(
        self,
    ) -> tuple[tuple[GameAction, tuple[str, ...]], ...]:
        return tuple(
            (action, self.card_names_for_action(action))
            for action in self.legal_actions
        )
