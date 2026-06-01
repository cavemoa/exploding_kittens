"""Action definitions for player turns."""

from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    DRAW = "draw"
    PLAY_SKIP = "play_skip"
    PLAY_ATTACK = "play_attack"
    PLAY_SHUFFLE = "play_shuffle"
    PLAY_FAVOR = "play_favor"
    PLAY_ALTER_THE_FUTURE = "play_alter_the_future"
    PLAY_SEE_THE_FUTURE = "play_see_the_future"
    PLAY_TWO_OF_A_KIND = "play_two_of_a_kind"


@dataclass(frozen=True)
class GameAction:
    kind: Action
    card_indexes: tuple[int, ...] = ()
    target_player: str | None = None
    named_card: str | None = None
    reorder: tuple[int, ...] = ()
