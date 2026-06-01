import random

import pytest

from exploding_kittens import (
    Action,
    AlterTheFutureCard,
    AttackCard,
    Card,
    DefuseCard,
    ExplodingKittenCard,
    FavorCard,
    GameAction,
    GameEngine,
    IllegalActionError,
    NormalCard,
    PlayerState,
    RandomAgent,
    SeeTheFutureCard,
    ShuffleCard,
    SkipCard,
    TwoOfAKindRule,
)


def engine_with(
    hands: list[list[Card]],
    deck_top_last: list[Card],
    combo_rules=(),
    reveal_opponent_card_counts: bool = False,
) -> GameEngine:
    players = [
        PlayerState(name=f"player_{index + 1}", hand=hand.copy())
        for index, hand in enumerate(hands)
    ]
    return GameEngine(
        players,
        deck_top_last,
        combo_rules=combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )


def seeded_engine_with(
    hands: list[list[Card]],
    deck_top_last: list[Card],
    seed: int,
) -> GameEngine:
    players = [
        PlayerState(name=f"player_{index + 1}", hand=hand.copy())
        for index, hand in enumerate(hands)
    ]
    return GameEngine(players, deck_top_last, rng=random.Random(seed))


def test_drawing_normal_card_adds_it_to_hand_and_advances_turn() -> None:
    engine = engine_with(
        hands=[[DefuseCard()], [DefuseCard()]],
        deck_top_last=[ExplodingKittenCard(), NormalCard()],
    )

    events = engine.step(Action.DRAW)

    assert events[0].kind == "draw"
    assert engine.players[0].hand == [DefuseCard(), NormalCard()]
    assert engine.current_player.name == "player_2"
    assert engine.turn_count == 1


def test_skip_discards_card_and_does_not_draw() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), SkipCard()], [DefuseCard()]],
        deck_top_last=[ExplodingKittenCard(), NormalCard()],
    )

    events = engine.step(GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,)))

    assert events[0].kind == "skip"
    assert engine.players[0].hand == [DefuseCard()]
    assert engine.discard_pile == [SkipCard()]
    assert engine.draw_pile == [ExplodingKittenCard(), NormalCard()]
    assert engine.current_player.name == "player_2"


def test_attack_discards_card_and_gives_next_player_two_turns() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), AttackCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), NormalCard(), NormalCard()],
    )

    events = engine.step(GameAction(kind=Action.PLAY_ATTACK, card_indexes=(1,)))

    assert events[0].kind == "attack"
    assert engine.players[0].hand == [DefuseCard()]
    assert engine.discard_pile == [AttackCard()]
    assert engine.current_player.name == "player_2"
    assert engine.current_turns_remaining == 2


def test_shuffle_discards_card_and_reorders_draw_pile() -> None:
    starting_deck = [
        NormalCard(),
        SkipCard(),
        AttackCard(),
        ExplodingKittenCard(),
    ]
    engine = seeded_engine_with(
        hands=[[DefuseCard(), ShuffleCard()], [DefuseCard()]],
        deck_top_last=starting_deck,
        seed=3,
    )

    events = engine.step(GameAction(kind=Action.PLAY_SHUFFLE, card_indexes=(1,)))

    assert events[0].kind == "shuffle"
    assert engine.players[0].hand == [DefuseCard()]
    assert engine.discard_pile == [ShuffleCard()]
    assert sorted(card.name for card in engine.draw_pile) == sorted(card.name for card in starting_deck)
    assert engine.draw_pile != starting_deck


def test_favor_discards_card_and_takes_random_card_from_next_player() -> None:
    engine = seeded_engine_with(
        hands=[
            [DefuseCard(), FavorCard()],
            [DefuseCard(), SkipCard(), AttackCard()],
        ],
        deck_top_last=[NormalCard()],
        seed=1,
    )

    action = GameAction(
        kind=Action.PLAY_FAVOR,
        card_indexes=(1,),
        target_player="player_2",
    )
    events = engine.step(action)

    assert events[0].kind == "favor"
    assert engine.players[0].hand == [DefuseCard(), DefuseCard()]
    assert engine.players[1].hand == [SkipCard(), AttackCard()]
    assert engine.discard_pile == [FavorCard()]
    assert engine.current_player.name == "player_2"


def test_favor_can_target_a_specific_player() -> None:
    engine = engine_with(
        hands=[
            [DefuseCard(), FavorCard()],
            [DefuseCard(), SkipCard()],
            [AttackCard()],
        ],
        deck_top_last=[NormalCard()],
    )

    action = GameAction(
        kind=Action.PLAY_FAVOR,
        card_indexes=(1,),
        target_player="player_3",
    )
    events = engine.step(action)

    assert events[0].kind == "favor"
    assert engine.players[0].hand == [DefuseCard(), AttackCard()]
    assert engine.players[1].hand == [DefuseCard(), SkipCard()]
    assert engine.players[2].hand == []


def test_favor_is_not_legal_without_an_opponent_card_to_take() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), FavorCard()], []],
        deck_top_last=[NormalCard()],
    )

    assert Action.PLAY_FAVOR not in {
        action.kind for action in engine.legal_actions("player_1")
    }


def test_alter_the_future_discards_card_and_reverses_top_three_cards() -> None:
    bottom_card = NormalCard()
    third_from_top = SkipCard()
    second_from_top = AttackCard()
    top_card = ExplodingKittenCard()
    engine = engine_with(
        hands=[[DefuseCard(), AlterTheFutureCard()], [DefuseCard()]],
        deck_top_last=[bottom_card, third_from_top, second_from_top, top_card],
    )

    events = engine.step(
        GameAction(kind=Action.PLAY_ALTER_THE_FUTURE, card_indexes=(1,))
    )

    assert events[0].kind == "alter_the_future"
    assert engine.players[0].hand == [DefuseCard()]
    assert engine.discard_pile == [AlterTheFutureCard()]
    assert engine.draw_pile == [bottom_card, top_card, second_from_top, third_from_top]


def test_see_the_future_stores_top_three_cards_for_current_player_only() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), SeeTheFutureCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), SkipCard(), AttackCard(), ExplodingKittenCard()],
    )

    events = engine.step(GameAction(kind=Action.PLAY_SEE_THE_FUTURE, card_indexes=(1,)))

    assert events[0].kind == "see_the_future"
    assert engine.players[0].hand == [DefuseCard()]
    assert engine.discard_pile == [SeeTheFutureCard()]
    assert engine.observe("player_1").known_top_cards == (
        "exploding_kitten",
        "attack",
        "skip",
    )
    assert engine.observe("player_2").known_top_cards == ()


def test_future_knowledge_is_cleared_when_draw_pile_changes() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), SeeTheFutureCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), SkipCard(), AttackCard(), NormalCard()],
    )

    engine.step(GameAction(kind=Action.PLAY_SEE_THE_FUTURE, card_indexes=(1,)))
    engine.step(Action.DRAW)

    assert engine.observe("player_1").known_top_cards == ()


def test_attacked_player_takes_two_turns_before_play_moves_on() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), AttackCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), NormalCard(), NormalCard()],
    )

    engine.step(GameAction(kind=Action.PLAY_ATTACK, card_indexes=(1,)))
    engine.step(Action.DRAW)

    assert engine.current_player.name == "player_2"
    assert engine.current_turns_remaining == 1

    engine.step(Action.DRAW)

    assert engine.current_player.name == "player_1"
    assert engine.current_turns_remaining == 1


def test_observation_contains_agent_view_of_game_state() -> None:
    engine = engine_with(
        hands=[[DefuseCard(), SkipCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), ExplodingKittenCard()],
    )
    engine.discard_pile.append(AttackCard())
    engine.current_turns_remaining = 2
    engine.next_player_turns = 1

    observation = engine.observe("player_1")

    assert observation.player == "player_1"
    assert observation.own_hand == ("defuse", "skip")
    assert observation.hand_counts["defuse"] == 1
    assert observation.hand_counts["skip"] == 1
    assert observation.hand_counts["attack"] == 0
    assert observation.discard_counts["attack"] == 1
    assert observation.draw_pile_size == 2
    assert observation.living_players == 2
    assert observation.current_turns_remaining == 2
    assert observation.next_player_turns == 1
    assert observation.known_top_cards == ()
    assert observation.opponent_card_counts == {}
    assert observation.public_events == ()
    assert observation.legal_actions == (
        GameAction(kind=Action.DRAW),
        GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,)),
    )
    assert observation.card_names_for_action(observation.legal_actions[0]) == ()
    assert observation.card_names_for_action(observation.legal_actions[1]) == ("skip",)
    assert observation.legal_actions_with_card_names() == (
        (GameAction(kind=Action.DRAW), ()),
        (GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,)), ("skip",)),
    )


def test_observation_card_names_help_agents_inspect_combo_actions() -> None:
    engine = engine_with(
        hands=[
            [NormalCard(), NormalCard(), SkipCard(), SkipCard()],
            [FavorCard()],
            [AttackCard()],
        ],
        deck_top_last=[NormalCard()],
        combo_rules=[TwoOfAKindRule()],
    )

    observation = engine.observe("player_1")
    skip_pair_action = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(2, 3),
        target_player="player_3",
    )

    assert observation.own_hand == ("normal", "normal", "skip", "skip")
    assert skip_pair_action in observation.legal_actions
    assert observation.card_names_for_action(skip_pair_action) == ("skip", "skip")
    assert all("favor" not in cards for _, cards in observation.legal_actions_with_card_names())


def test_observation_can_reveal_opponent_card_counts_when_enabled() -> None:
    engine = engine_with(
        hands=[[DefuseCard()], [DefuseCard(), SkipCard()], [AttackCard()]],
        deck_top_last=[NormalCard()],
        reveal_opponent_card_counts=True,
    )

    observation = engine.observe("player_1")

    assert observation.opponent_card_counts == {
        "player_2": 2,
        "player_3": 1,
    }


def test_exploding_kitten_without_defuse_eliminates_player() -> None:
    engine = engine_with(
        hands=[[NormalCard()], [DefuseCard()]],
        deck_top_last=[ExplodingKittenCard()],
    )

    events = engine.step(Action.DRAW)

    assert events[0].kind == "eliminated"
    assert not engine.players[0].alive
    assert engine.is_over
    assert engine.winner == "player_2"


def test_exploding_kitten_with_defuse_consumes_defuse_and_returns_kitten_to_bottom() -> None:
    engine = engine_with(
        hands=[[DefuseCard()], [DefuseCard()]],
        deck_top_last=[NormalCard(), ExplodingKittenCard()],
    )

    events = engine.step(Action.DRAW)

    assert events[0].kind == "defuse"
    assert engine.players[0].hand == []
    assert engine.discard_pile == [DefuseCard()]
    assert engine.draw_pile == [ExplodingKittenCard(), NormalCard()]
    assert engine.current_player.name == "player_2"


def test_legal_actions_only_include_skip_when_current_player_has_skip() -> None:
    engine = engine_with(
        hands=[
            [
                DefuseCard(),
                SkipCard(),
                AttackCard(),
                ShuffleCard(),
                FavorCard(),
                AlterTheFutureCard(),
                SeeTheFutureCard(),
            ],
            [DefuseCard()],
        ],
        deck_top_last=[NormalCard()],
    )

    assert engine.legal_actions("player_1") == (
        GameAction(kind=Action.DRAW),
        GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,)),
        GameAction(kind=Action.PLAY_ATTACK, card_indexes=(2,)),
        GameAction(kind=Action.PLAY_SHUFFLE, card_indexes=(3,)),
        GameAction(kind=Action.PLAY_FAVOR, card_indexes=(4,), target_player="player_2"),
        GameAction(kind=Action.PLAY_ALTER_THE_FUTURE, card_indexes=(5,)),
        GameAction(kind=Action.PLAY_SEE_THE_FUTURE, card_indexes=(6,)),
    )
    assert engine.legal_actions("player_2") == ()


def test_illegal_skip_raises_error() -> None:
    engine = engine_with(
        hands=[[DefuseCard()], [DefuseCard()]],
        deck_top_last=[NormalCard()],
    )

    with pytest.raises(IllegalActionError):
        engine.step(GameAction(kind=Action.PLAY_SKIP, card_indexes=(0,)))


def test_two_of_a_kind_is_only_legal_when_combo_rule_is_enabled() -> None:
    engine = engine_with(
        hands=[[NormalCard(), NormalCard()], [SkipCard()]],
        deck_top_last=[NormalCard()],
    )

    assert Action.PLAY_TWO_OF_A_KIND not in {
        action.kind for action in engine.legal_actions("player_1")
    }


def test_two_of_a_kind_discards_pair_and_steals_random_card() -> None:
    engine = seeded_engine_with(
        hands=[[NormalCard(), NormalCard()], [SkipCard(), AttackCard()]],
        deck_top_last=[NormalCard()],
        seed=1,
    )
    engine.combo_rules = [TwoOfAKindRule()]

    action = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(0, 1),
        target_player="player_2",
    )
    events = engine.step(action)

    assert events[0].kind == "two_of_a_kind"
    assert engine.players[0].hand == [SkipCard()]
    assert engine.players[1].hand == [AttackCard()]
    assert engine.discard_pile == [NormalCard(), NormalCard()]
    assert engine.current_player.name == "player_2"


def test_two_of_a_kind_can_choose_pair_and_target() -> None:
    engine = engine_with(
        hands=[
            [NormalCard(), NormalCard(), SkipCard(), SkipCard()],
            [FavorCard()],
            [AttackCard()],
        ],
        deck_top_last=[NormalCard()],
        combo_rules=[TwoOfAKindRule()],
    )

    action = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(2, 3),
        target_player="player_3",
    )
    events = engine.step(action)

    assert events[0].kind == "two_of_a_kind"
    assert engine.players[0].hand == [NormalCard(), NormalCard(), AttackCard()]
    assert engine.players[1].hand == [FavorCard()]
    assert engine.players[2].hand == []
    assert engine.discard_pile == [SkipCard(), SkipCard()]


def test_two_of_a_kind_is_not_legal_without_an_opponent_card_to_steal() -> None:
    engine = engine_with(
        hands=[[NormalCard(), NormalCard()], []],
        deck_top_last=[NormalCard()],
        combo_rules=[TwoOfAKindRule()],
    )

    assert Action.PLAY_TWO_OF_A_KIND not in {
        action.kind for action in engine.legal_actions("player_1")
    }


def test_seeded_random_agents_finish_games() -> None:
    rng = random.Random(42)

    for game_index in range(50):
        engine = GameEngine.new_game(["player_1", "player_2"], seed=game_index)
        agents = {
            "player_1": RandomAgent(),
            "player_2": RandomAgent(),
        }

        while not engine.is_over and engine.turn_count < 500:
            player_name = engine.current_player.name
            action = agents[player_name].choose_action(engine.observe(player_name), rng)
            engine.step(action)

        assert engine.is_over
        assert engine.winner in {"player_1", "player_2"}


def test_new_game_can_exclude_attack_cards() -> None:
    engine = GameEngine.new_game(
        ["player_1", "player_2"],
        seed=1,
        exclude_cards=["attack"],
    )
    all_cards = [
        card
        for player in engine.players
        for card in player.hand
    ] + engine.draw_pile

    assert not any(isinstance(card, AttackCard) for card in all_cards)


def test_new_game_can_include_specific_optional_cards() -> None:
    engine = GameEngine.new_game(
        ["player_1", "player_2"],
        seed=1,
        include_cards=["normal", "skip"],
    )
    all_cards = [
        card
        for player in engine.players
        for card in player.hand
    ] + engine.draw_pile

    assert not any(isinstance(card, AttackCard) for card in all_cards)
    assert not any(isinstance(card, ShuffleCard) for card in all_cards)
    assert not any(isinstance(card, FavorCard) for card in all_cards)
    assert not any(isinstance(card, AlterTheFutureCard) for card in all_cards)
    assert not any(isinstance(card, SeeTheFutureCard) for card in all_cards)
    assert any(isinstance(card, SkipCard) for card in all_cards)
