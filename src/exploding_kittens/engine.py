"""Pure Python game engine for a small Exploding Kittens simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Iterable

from .actions import Action, GameAction
from .cards import (
    CARD_TYPES,
    DECK_CARD_REGISTRY,
    DECK_CARD_TYPES,
    Card,
    DefuseCard,
    ExplodingKittenCard,
)
from .combo_rules import COMBO_RULE_REGISTRY, ComboRule
from .observations import GameObservation


@dataclass
class PlayerState:
    name: str
    hand: list[Card] = field(default_factory=list)
    alive: bool = True
    known_top_cards: tuple[str, ...] = ()

    def has_card(self, card_type: type[Card]) -> bool:
        return self.find_card(card_type) is not None

    def find_card(self, card_type: type[Card]) -> Card | None:
        return next((card for card in self.hand if isinstance(card, card_type)), None)

    def remove_card_type(self, card_type: type[Card]) -> Card:
        card = self.find_card(card_type)
        if card is None:
            raise ValueError(f"{self.name} does not have a {card_type.__name__}.")
        self.hand.remove(card)
        return card

    def remove_card_at(self, card_index: int) -> Card:
        return self.hand.pop(card_index)

    def remove_cards_at(self, card_indexes: Iterable[int]) -> tuple[Card, ...]:
        removed_cards: list[Card] = []
        for card_index in sorted(card_indexes, reverse=True):
            removed_cards.append(self.hand.pop(card_index))
        return tuple(reversed(removed_cards))


@dataclass(frozen=True)
class GameEvent:
    kind: str
    player: str
    message: str
    card: Card | None = None


class IllegalActionError(ValueError):
    """Raised when a player attempts an action that is not currently legal."""


class GameEngine:
    """A deterministic, testable rules engine.

    The top of the draw pile is the end of the list so drawing can use pop().
    """

    def __init__(
        self,
        players: Iterable[PlayerState],
        draw_pile: Iterable[Card],
        *,
        starting_player_index: int = 0,
        rng: random.Random | None = None,
        combo_rules: Iterable[ComboRule] = (),
        reveal_opponent_card_counts: bool = False,
    ) -> None:
        self.players = list(players)
        if len(self.players) < 2:
            raise ValueError("At least two players are required.")

        self.draw_pile = list(draw_pile)
        self.discard_pile: list[Card] = []
        self.current_player_index = starting_player_index
        self.current_turns_remaining = 1
        self.next_player_turns = 1
        self.turn_count = 0
        self.events: list[GameEvent] = []
        self.rng = rng or random.Random()
        self.combo_rules = list(combo_rules)
        self.reveal_opponent_card_counts = reveal_opponent_card_counts

        if not 0 <= self.current_player_index < len(self.players):
            raise ValueError("starting_player_index is out of range.")
        if not self.current_player.alive:
            self._advance_to_next_alive_player()

    @classmethod
    def new_game(
        cls,
        player_names: Iterable[str],
        *,
        seed: int | None = None,
        starting_hand_size: int = 4,
        normal_cards: int | None = None,
        skip_cards: int | None = None,
        attack_cards: int | None = None,
        shuffle_cards: int | None = None,
        favor_cards: int | None = None,
        alter_the_future_cards: int | None = None,
        see_the_future_cards: int | None = None,
        include_cards: Iterable[str] | None = None,
        exclude_cards: Iterable[str] = (),
        enabled_combo_rules: Iterable[str] = (),
        reveal_opponent_card_counts: bool = False,
    ) -> GameEngine:
        names = list(player_names)
        if len(names) < 2:
            raise ValueError("At least two player names are required.")
        if starting_hand_size < 1:
            raise ValueError("starting_hand_size must include at least the defuse card.")

        rng = random.Random(seed)
        normal_count = normal_cards if normal_cards is not None else len(names) * 8
        skip_count = skip_cards if skip_cards is not None else len(names) * 2
        attack_count = attack_cards if attack_cards is not None else len(names)
        shuffle_count = shuffle_cards if shuffle_cards is not None else len(names)
        favor_count = favor_cards if favor_cards is not None else len(names)
        alter_the_future_count = (
            alter_the_future_cards if alter_the_future_cards is not None else len(names)
        )
        see_the_future_count = (
            see_the_future_cards if see_the_future_cards is not None else len(names)
        )
        card_counts = {
            "normal": normal_count,
            "skip": skip_count,
            "attack": attack_count,
            "shuffle": shuffle_count,
            "favor": favor_count,
            "alter_the_future": alter_the_future_count,
            "see_the_future": see_the_future_count,
        }
        deck_card_types = cls._select_deck_card_types(include_cards, exclude_cards)
        deal_pool = [
            card_type()
            for card_type in deck_card_types
            for _ in range(card_counts[card_type.name])
        ]
        rng.shuffle(deal_pool)

        players = [PlayerState(name=name, hand=[DefuseCard()]) for name in names]
        cards_to_deal = (starting_hand_size - 1) * len(players)
        if len(deal_pool) < cards_to_deal:
            raise ValueError("Not enough non-kitten cards to deal starting hands.")

        for _ in range(starting_hand_size - 1):
            for player in players:
                player.hand.append(deal_pool.pop())

        deck = deal_pool + [ExplodingKittenCard() for _ in range(len(players) - 1)]
        rng.shuffle(deck)
        combo_rules = cls._select_combo_rules(enabled_combo_rules)
        return cls(
            players,
            deck,
            rng=rng,
            combo_rules=combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
        )

    @staticmethod
    def _select_deck_card_types(
        include_cards: Iterable[str] | None,
        exclude_cards: Iterable[str],
    ) -> tuple[type[Card], ...]:
        if include_cards is None:
            selected = set(DECK_CARD_REGISTRY)
        else:
            selected = set(include_cards)

        excluded = set(exclude_cards)
        known_cards = set(DECK_CARD_REGISTRY)
        unknown_cards = sorted((selected | excluded) - known_cards)
        if unknown_cards:
            known = ", ".join(sorted(known_cards))
            unknown = ", ".join(unknown_cards)
            raise ValueError(f"Unknown optional card(s): {unknown}. Known optional cards: {known}.")

        selected -= excluded
        return tuple(card_type for card_type in DECK_CARD_TYPES if card_type.name in selected)

    @staticmethod
    def _select_combo_rules(enabled_combo_rules: Iterable[str]) -> tuple[ComboRule, ...]:
        selected = set(enabled_combo_rules)
        known_rules = set(COMBO_RULE_REGISTRY)
        unknown_rules = sorted(selected - known_rules)
        if unknown_rules:
            known = ", ".join(sorted(known_rules))
            unknown = ", ".join(unknown_rules)
            raise ValueError(f"Unknown combo rule(s): {unknown}. Known combo rules: {known}.")
        return tuple(
            rule_type()
            for rule_name, rule_type in COMBO_RULE_REGISTRY.items()
            if rule_name in selected
        )

    @property
    def current_player(self) -> PlayerState:
        return self.players[self.current_player_index]

    @property
    def alive_players(self) -> list[PlayerState]:
        return [player for player in self.players if player.alive]

    @property
    def is_over(self) -> bool:
        return len(self.alive_players) <= 1

    @property
    def winner(self) -> str | None:
        alive = self.alive_players
        if len(alive) == 1:
            return alive[0].name
        return None

    def legal_actions(self, player_name: str | None = None) -> tuple[GameAction, ...]:
        player = self._player_by_name(player_name) if player_name else self.current_player
        if self.is_over or player is not self.current_player or not player.alive:
            return ()

        actions = [GameAction(kind=Action.DRAW)]
        for card_index, card in enumerate(player.hand):
            actions.extend(card.legal_actions(self, player, card_index))
        for combo_rule in self.combo_rules:
            actions.extend(combo_rule.legal_actions(self, player))
        return tuple(actions)

    def hand_counts(self, player_name: str) -> dict[str, int]:
        player = self._player_by_name(player_name)
        return {
            card_type.name: sum(isinstance(card, card_type) for card in player.hand)
            for card_type in CARD_TYPES
        }

    def discard_counts(self) -> dict[str, int]:
        return {
            card_type.name: sum(isinstance(card, card_type) for card in self.discard_pile)
            for card_type in CARD_TYPES
        }

    def opponent_card_counts(self, player: PlayerState) -> dict[str, int]:
        if not self.reveal_opponent_card_counts:
            return {}
        return {
            opponent.name: len(opponent.hand)
            for opponent in self.players
            if opponent is not player and opponent.alive
        }

    def observe(self, player_name: str | None = None) -> GameObservation:
        player = self._player_by_name(player_name) if player_name else self.current_player
        return GameObservation(
            player=player.name,
            own_hand=tuple(card.name for card in player.hand),
            hand_counts=self.hand_counts(player.name),
            discard_counts=self.discard_counts(),
            draw_pile_size=len(self.draw_pile),
            living_players=len(self.alive_players),
            current_turns_remaining=self.current_turns_remaining,
            next_player_turns=self.next_player_turns,
            known_top_cards=player.known_top_cards,
            opponent_card_counts=self.opponent_card_counts(player),
            public_events=tuple(event.kind for event in self.events),
            legal_actions=self.legal_actions(player.name),
        )

    def create_event(
        self,
        *,
        kind: str,
        player: PlayerState,
        message: str,
        card: Card | None = None,
    ) -> GameEvent:
        return GameEvent(kind=kind, player=player.name, card=card, message=message)

    def attack_next_player(self, turns: int) -> None:
        if turns < 1:
            raise ValueError("Attack must assign at least one turn.")
        self.current_turns_remaining = 0
        self.next_player_turns = turns

    def find_favor_target(self, player: PlayerState) -> PlayerState | None:
        return self.find_steal_target(player)

    def find_steal_target(self, player: PlayerState) -> PlayerState | None:
        targets = self.find_steal_targets(player)
        return targets[0] if targets else None

    def find_steal_targets(self, player: PlayerState) -> tuple[PlayerState, ...]:
        player_index = next(
            index for index, candidate in enumerate(self.players) if candidate is player
        )
        targets = []
        for offset in range(1, len(self.players)):
            target = self.players[(player_index + offset) % len(self.players)]
            if target.alive and target.hand:
                targets.append(target)
        return tuple(targets)

    def player_by_name(self, player_name: str | None) -> PlayerState | None:
        if player_name is None:
            return None
        try:
            return self._player_by_name(player_name)
        except KeyError:
            return None

    def take_random_card(self, player: PlayerState) -> Card:
        if not player.hand:
            raise ValueError(f"{player.name} has no cards to take.")
        index = self.rng.randrange(len(player.hand))
        return player.hand.pop(index)

    def reverse_top_cards(self, count: int) -> None:
        if count < 1:
            raise ValueError("count must be at least 1.")
        cards_to_reorder = min(count, len(self.draw_pile))
        if cards_to_reorder < 2:
            return
        self.draw_pile[-cards_to_reorder:] = reversed(self.draw_pile[-cards_to_reorder:])
        self.clear_future_knowledge()

    def peek_top_cards(self, count: int) -> tuple[str, ...]:
        if count < 1:
            raise ValueError("count must be at least 1.")
        return tuple(card.name for card in reversed(self.draw_pile[-count:]))

    def clear_future_knowledge(self) -> None:
        for player in self.players:
            player.known_top_cards = ()

    def step(self, action: GameAction | Action | str) -> list[GameEvent]:
        if self.is_over:
            return []

        action = self._coerce_action(action)

        if action not in self.legal_actions():
            raise IllegalActionError(
                f"{self.current_player.name} cannot perform {action.kind.value}."
            )

        if action.kind is Action.DRAW:
            events = self._draw_for_current_player()
        else:
            playable_card = self._find_card_for_action(self.current_player, action)
            if playable_card is not None:
                events = playable_card.play(self, self.current_player, action)
            else:
                combo_rule = self._find_combo_rule_for_action(self.current_player, action)
                if combo_rule is None:
                    raise IllegalActionError(
                        f"{self.current_player.name} cannot perform {action.kind.value}."
                    )
                events = combo_rule.play(self, self.current_player, action)

        self.turn_count += 1
        self.events.extend(events)
        if not self.is_over:
            self._complete_current_turn()
        return events

    def _draw_for_current_player(self) -> list[GameEvent]:
        player = self.current_player
        if not self.draw_pile:
            raise IllegalActionError("Cannot draw from an empty draw pile.")

        card = self.draw_pile.pop()
        self.clear_future_knowledge()
        return card.on_draw(self, player)

    def _advance_to_next_alive_player(self) -> None:
        for offset in range(1, len(self.players) + 1):
            next_index = (self.current_player_index + offset) % len(self.players)
            if self.players[next_index].alive:
                self.current_player_index = next_index
                return

    def _complete_current_turn(self) -> None:
        if self.current_turns_remaining > 0:
            self.current_turns_remaining -= 1

        if self.current_turns_remaining > 0 and self.current_player.alive:
            return

        self._advance_to_next_alive_player()
        self.current_turns_remaining = self.next_player_turns
        self.next_player_turns = 1

    def _player_by_name(self, player_name: str) -> PlayerState:
        for player in self.players:
            if player.name == player_name:
                return player
        raise KeyError(f"Unknown player: {player_name}")

    def _coerce_action(self, action: GameAction | Action | str) -> GameAction:
        if isinstance(action, GameAction):
            return action

        action_kind = Action(action)
        if action_kind is Action.DRAW:
            return GameAction(kind=Action.DRAW)
        return GameAction(kind=action_kind)

    def _find_card_for_action(self, player: PlayerState, action: GameAction) -> Card | None:
        if len(action.card_indexes) != 1:
            return None
        card_index = action.card_indexes[0]
        if not 0 <= card_index < len(player.hand):
            return None
        card = player.hand[card_index]
        if card.action is not action.kind:
            return None
        return card

    def _find_combo_rule_for_action(
        self,
        player: PlayerState,
        action: GameAction,
    ) -> ComboRule | None:
        return next(
            (
                combo_rule
                for combo_rule in self.combo_rules
                if combo_rule.action is action.kind and action in combo_rule.legal_actions(self, player)
            ),
            None,
        )
