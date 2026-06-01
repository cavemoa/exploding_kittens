"""Multi-card combo rules that are separate from individual card effects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, ClassVar

from .actions import Action, GameAction
from .cards import Card

if TYPE_CHECKING:
    from .engine import GameEngine, GameEvent, PlayerState


@dataclass(frozen=True)
class ComboPlay:
    cards: tuple[Card, ...]
    target: PlayerState


class ComboRule:
    name: ClassVar[str] = "combo"
    action: ClassVar[Action]

    def find_play(self, engine: GameEngine, player: PlayerState) -> ComboPlay | None:
        raise NotImplementedError

    def can_play(self, engine: GameEngine, player: PlayerState) -> bool:
        return self.find_play(engine, player) is not None

    def legal_actions(self, engine: GameEngine, player: PlayerState) -> tuple[GameAction, ...]:
        play = self.find_play(engine, player)
        if play is None:
            return ()
        return (
            GameAction(
                kind=self.action,
                card_indexes=tuple(
                    card_index
                    for card in play.cards
                    for card_index, hand_card in enumerate(player.hand)
                    if hand_card is card
                ),
                target_player=play.target.name,
            ),
        )

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        raise NotImplementedError


class TwoOfAKindRule(ComboRule):
    name = "two_of_a_kind"
    action = Action.PLAY_TWO_OF_A_KIND

    def find_play(self, engine: GameEngine, player: PlayerState) -> ComboPlay | None:
        targets = engine.find_steal_targets(player)
        if not targets:
            return None

        cards_by_name: dict[str, list[Card]] = defaultdict(list)
        for card in player.hand:
            cards_by_name[card.name].append(card)

        for cards in cards_by_name.values():
            if len(cards) >= 2:
                return ComboPlay(cards=(cards[0], cards[1]), target=targets[0])
        return None

    def legal_actions(self, engine: GameEngine, player: PlayerState) -> tuple[GameAction, ...]:
        targets = engine.find_steal_targets(player)
        if not targets:
            return ()

        indexes_by_name: dict[str, list[int]] = defaultdict(list)
        for card_index, card in enumerate(player.hand):
            indexes_by_name[card.name].append(card_index)

        actions = []
        for indexes in indexes_by_name.values():
            for pair_indexes in combinations(indexes, 2):
                actions.extend(
                    GameAction(
                        kind=self.action,
                        card_indexes=pair_indexes,
                        target_player=target.name,
                    )
                    for target in targets
                )
        return tuple(actions)

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        if action not in self.legal_actions(engine, player):
            raise ValueError(f"{player.name} cannot play two of a kind.")

        target = engine.player_by_name(action.target_player)
        if target is None:
            raise ValueError(f"{player.name} cannot play two of a kind without a target.")

        played_cards = player.remove_cards_at(action.card_indexes)
        engine.discard_pile.extend(played_cards)

        stolen_card = engine.take_random_card(target)
        player.hand.append(stolen_card)
        pair_name = played_cards[0].name
        return [
            engine.create_event(
                kind=self.name,
                player=player,
                card=stolen_card,
                message=(
                    f"{player.name} played two {pair_name} cards on "
                    f"{target.name} and stole {stolen_card.name}."
                ),
            )
        ]


COMBO_RULE_TYPES: tuple[type[ComboRule], ...] = (TwoOfAKindRule,)

COMBO_RULE_REGISTRY: dict[str, type[ComboRule]] = {
    rule_type.name: rule_type for rule_type in COMBO_RULE_TYPES
}
