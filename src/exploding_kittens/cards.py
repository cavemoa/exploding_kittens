"""Card classes and their rule effects."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from .actions import Action, GameAction

if TYPE_CHECKING:
    from .engine import GameEngine, GameEvent, PlayerState


class Card:
    """Base card type.

    Cards own their local rule effect. The engine still owns turn order,
    validation, and shared state such as draw/discard piles.
    """

    name: ClassVar[str] = "card"
    playable: ClassVar[bool] = False
    action: ClassVar[Action | None] = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other)

    def __hash__(self) -> int:
        return hash(type(self))

    def can_play(self, engine: GameEngine, player: PlayerState) -> bool:
        return self.playable

    def legal_actions(
        self,
        engine: GameEngine,
        player: PlayerState,
        card_index: int,
    ) -> tuple[GameAction, ...]:
        if self.action is None or not self.can_play(engine, player):
            return ()
        return (GameAction(kind=self.action, card_indexes=(card_index,)),)

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        raise NotImplementedError(f"{type(self).__name__} cannot be played.")

    def on_draw(self, engine: GameEngine, player: PlayerState) -> list[GameEvent]:
        player.hand.append(self)
        return [
            engine.create_event(
                kind="draw",
                player=player,
                card=self,
                message=f"{player.name} drew {self.name}.",
            )
        ]


class NormalCard(Card):
    name = "normal"


class SkipCard(Card):
    name = "skip"
    playable = True
    action = Action.PLAY_SKIP

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        return [
            engine.create_event(
                kind="skip",
                player=player,
                card=played_card,
                message=f"{player.name} played skip.",
            )
        ]


class AttackCard(Card):
    name = "attack"
    playable = True
    action = Action.PLAY_ATTACK

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        engine.attack_next_player(turns=2)
        return [
            engine.create_event(
                kind="attack",
                player=player,
                card=played_card,
                message=f"{player.name} played attack.",
            )
        ]


class ShuffleCard(Card):
    name = "shuffle"
    playable = True
    action = Action.PLAY_SHUFFLE

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        engine.rng.shuffle(engine.draw_pile)
        engine.clear_future_knowledge()
        return [
            engine.create_event(
                kind="shuffle",
                player=player,
                card=played_card,
                message=f"{player.name} played shuffle.",
            )
        ]


class FavorCard(Card):
    name = "favor"
    playable = True
    action = Action.PLAY_FAVOR

    def can_play(self, engine: GameEngine, player: PlayerState) -> bool:
        return self.playable and bool(engine.find_steal_targets(player))

    def legal_actions(
        self,
        engine: GameEngine,
        player: PlayerState,
        card_index: int,
    ) -> tuple[GameAction, ...]:
        if self.action is None or not self.can_play(engine, player):
            return ()
        return tuple(
            GameAction(
                kind=self.action,
                card_indexes=(card_index,),
                target_player=target.name,
            )
            for target in engine.find_steal_targets(player)
        )

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        target = engine.player_by_name(action.target_player)
        if target is None or target not in engine.find_steal_targets(player):
            raise ValueError(f"{player.name} has no valid favor target.")

        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        transferred_card = engine.take_random_card(target)
        player.hand.append(transferred_card)
        return [
            engine.create_event(
                kind="favor",
                player=player,
                card=played_card,
                message=(
                    f"{player.name} played favor on {target.name} "
                    f"and received {transferred_card.name}."
                ),
            )
        ]


class AlterTheFutureCard(Card):
    name = "alter_the_future"
    playable = True
    action = Action.PLAY_ALTER_THE_FUTURE

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        engine.reverse_top_cards(count=3)
        return [
            engine.create_event(
                kind="alter_the_future",
                player=player,
                card=played_card,
                message=f"{player.name} altered the future.",
            )
        ]


class SeeTheFutureCard(Card):
    name = "see_the_future"
    playable = True
    action = Action.PLAY_SEE_THE_FUTURE

    def play(
        self,
        engine: GameEngine,
        player: PlayerState,
        action: GameAction,
    ) -> list[GameEvent]:
        played_card = player.remove_card_at(action.card_indexes[0])
        engine.discard_pile.append(played_card)
        player.known_top_cards = engine.peek_top_cards(count=3)
        return [
            engine.create_event(
                kind="see_the_future",
                player=player,
                card=played_card,
                message=f"{player.name} saw the future.",
            )
        ]


class DefuseCard(Card):
    name = "defuse"

    def use_to_defuse(
        self,
        engine: GameEngine,
        player: PlayerState,
        exploding_card: ExplodingKittenCard,
    ) -> list[GameEvent]:
        used_card = player.remove_card_type(type(self))
        engine.discard_pile.append(used_card)
        engine.draw_pile.insert(0, exploding_card)
        engine.clear_future_knowledge()
        return [
            engine.create_event(
                kind="defuse",
                player=player,
                card=exploding_card,
                message=f"{player.name} defused an exploding kitten.",
            )
        ]


class ExplodingKittenCard(Card):
    name = "exploding_kitten"

    def on_draw(self, engine: GameEngine, player: PlayerState) -> list[GameEvent]:
        defuse_card = player.find_card(DefuseCard)
        if isinstance(defuse_card, DefuseCard):
            return defuse_card.use_to_defuse(engine, player, self)

        player.alive = False
        engine.discard_pile.append(self)
        return [
            engine.create_event(
                kind="eliminated",
                player=player,
                card=self,
                message=f"{player.name} exploded and is out.",
            )
        ]


CARD_TYPES: tuple[type[Card], ...] = (
    NormalCard,
    SkipCard,
    AttackCard,
    ShuffleCard,
    FavorCard,
    AlterTheFutureCard,
    SeeTheFutureCard,
    DefuseCard,
    ExplodingKittenCard,
)

DECK_CARD_TYPES: tuple[type[Card], ...] = (
    NormalCard,
    SkipCard,
    AttackCard,
    ShuffleCard,
    FavorCard,
    AlterTheFutureCard,
    SeeTheFutureCard,
)

DECK_CARD_REGISTRY: dict[str, type[Card]] = {
    card_type.name: card_type for card_type in DECK_CARD_TYPES
}
