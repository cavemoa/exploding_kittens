import random

from exploding_kittens import Action, GameAction, GameObservation, SafeRuleAgent


def observation_with(
    *,
    own_hand: tuple[str, ...],
    legal_actions: tuple[GameAction, ...],
    known_top_cards: tuple[str, ...] = (),
) -> GameObservation:
    return GameObservation(
        player="player_1",
        own_hand=own_hand,
        hand_counts={},
        discard_counts={},
        draw_pile_size=10,
        living_players=2,
        current_turns_remaining=1,
        next_player_turns=1,
        known_top_cards=known_top_cards,
        opponent_card_counts={},
        public_events=(),
        legal_actions=legal_actions,
    )


def test_safe_rule_agent_avoids_drawing_when_top_card_is_known_kitten() -> None:
    draw = GameAction(kind=Action.DRAW)
    skip = GameAction(kind=Action.PLAY_SKIP, card_indexes=(0,))
    observation = observation_with(
        own_hand=("skip",),
        legal_actions=(draw, skip),
        known_top_cards=("exploding_kitten", "normal", "normal"),
    )

    action = SafeRuleAgent().choose_action(observation, random.Random(1))

    assert action == skip


def test_safe_rule_agent_prefers_priority_responses_when_drawing_is_dangerous() -> None:
    draw = GameAction(kind=Action.DRAW)
    attack = GameAction(kind=Action.PLAY_ATTACK, card_indexes=(0,))
    skip = GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,))
    shuffle = GameAction(kind=Action.PLAY_SHUFFLE, card_indexes=(2,))
    alter = GameAction(kind=Action.PLAY_ALTER_THE_FUTURE, card_indexes=(3,))
    observation = observation_with(
        own_hand=("attack", "skip", "shuffle", "alter_the_future"),
        legal_actions=(draw, alter, attack, skip, shuffle),
        known_top_cards=("exploding_kitten",),
    )

    action = SafeRuleAgent().choose_action(observation, random.Random(1))

    assert action == shuffle


def test_safe_rule_agent_avoids_spending_defuse_in_two_of_a_kind() -> None:
    draw = GameAction(kind=Action.DRAW)
    defuse_pair = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(0, 1),
        target_player="player_2",
    )
    observation = observation_with(
        own_hand=("defuse", "defuse"),
        legal_actions=(draw, defuse_pair),
    )

    action = SafeRuleAgent().choose_action(observation, random.Random(1))

    assert action == draw


def test_safe_rule_agent_prefers_low_value_two_of_a_kind_pair() -> None:
    draw = GameAction(kind=Action.DRAW)
    skip_pair = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(0, 1),
        target_player="player_2",
    )
    normal_pair = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(2, 3),
        target_player="player_2",
    )
    observation = observation_with(
        own_hand=("skip", "skip", "normal", "normal"),
        legal_actions=(draw, skip_pair, normal_pair),
    )

    action = SafeRuleAgent().choose_action(observation, random.Random(1))

    assert action == normal_pair
