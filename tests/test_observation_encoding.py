import numpy as np

from exploding_kittens import (
    Action,
    AttackCard,
    DefuseCard,
    FavorCard,
    GameAction,
    GameEngine,
    NormalCard,
    ObservationEncoder,
    PlayerState,
    SeeTheFutureCard,
    ShuffleCard,
    SkipCard,
)


def observation_with(
    *,
    own_hand: tuple[str, ...] = ("defuse", "skip"),
    known_top_cards: tuple[str, ...] = (),
    opponent_card_counts: dict[str, int] | None = None,
    legal_actions: tuple[GameAction, ...] = (
        GameAction(kind=Action.DRAW),
        GameAction(kind=Action.PLAY_SKIP, card_indexes=(1,)),
    ),
):
    engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), SkipCard()]),
            PlayerState("player_2", [DefuseCard(), ShuffleCard()]),
        ],
        [NormalCard(), SeeTheFutureCard()],
    )
    observation = engine.observe("player_1")
    return type(observation)(
        player=observation.player,
        own_hand=own_hand,
        hand_counts=observation.hand_counts,
        discard_counts=observation.discard_counts,
        draw_pile_size=observation.draw_pile_size,
        living_players=observation.living_players,
        current_turns_remaining=observation.current_turns_remaining,
        next_player_turns=observation.next_player_turns,
        known_top_cards=known_top_cards,
        opponent_card_counts=opponent_card_counts or {},
        public_events=observation.public_events,
        legal_actions=legal_actions,
    )


def test_observation_encoder_returns_stable_shapes() -> None:
    encoder = ObservationEncoder(max_hand_size=4, max_players=3)
    observation = observation_with()

    encoded = encoder.encode(observation)

    assert encoded["observation"].shape == (encoder.observation_size,)
    assert encoded["observation"].dtype == np.float32
    assert encoded["action_mask"].shape == (encoder.action_encoder.action_space_size,)
    assert encoded["action_mask"].dtype == np.int8


def test_observation_encoder_hides_opponent_card_counts_by_default() -> None:
    encoder = ObservationEncoder(max_hand_size=4, max_players=3)
    hidden = observation_with(opponent_card_counts={})
    revealed_in_source = observation_with(
        opponent_card_counts={"player_2": 2, "player_3": 7}
    )

    np.testing.assert_array_equal(
        encoder.encode_observation(hidden),
        encoder.encode_observation(revealed_in_source),
    )


def test_observation_encoder_hides_exact_opponent_hands() -> None:
    encoder = ObservationEncoder(max_hand_size=4, max_players=2)
    first_engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), FavorCard()]),
            PlayerState("player_2", [DefuseCard(), ShuffleCard()]),
        ],
        [NormalCard()],
    )
    second_engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), FavorCard()]),
            PlayerState("player_2", [AttackCard(), NormalCard()]),
        ],
        [NormalCard()],
    )

    np.testing.assert_array_equal(
        encoder.encode_observation(first_engine.observe("player_1")),
        encoder.encode_observation(second_engine.observe("player_1")),
    )


def test_observation_encoder_can_include_opponent_card_counts_when_configured() -> None:
    encoder = ObservationEncoder(
        max_hand_size=8,
        max_players=3,
        include_opponent_card_counts=True,
    )
    observation = observation_with(opponent_card_counts={"player_2": 4})

    encoded = encoder.encode_observation(observation)

    assert encoded[-2:].tolist() == [0.5, 0.0]


def test_known_top_cards_are_encoded_only_when_present() -> None:
    encoder = ObservationEncoder(max_hand_size=4, max_players=3)
    hidden_future = encoder.encode_observation(observation_with())
    known_future = encoder.encode_observation(
        observation_with(known_top_cards=("exploding_kitten", "normal", "skip"))
    )
    known_start = (
        encoder.own_hand_size
        + encoder.hand_counts_size
        + encoder.discard_counts_size
        + encoder.scalar_size
    )
    exploding_index = encoder.card_index_by_name["exploding_kitten"]

    assert hidden_future[known_start] == 1.0
    assert known_future[known_start] == 0.0
    assert known_future[known_start + exploding_index] == 1.0


def test_action_mask_matches_engine_legal_actions() -> None:
    engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), SkipCard()]),
            PlayerState("player_2", [DefuseCard()]),
        ],
        [NormalCard()],
    )
    observation = engine.observe("player_1")
    encoder = ObservationEncoder(max_hand_size=4, max_players=2)

    encoded = encoder.encode(observation)
    expected_ids = {
        encoder.action_encoder.encode(action, observation)
        for action in engine.legal_actions("player_1")
    }

    assert encoded["action_mask"].sum() == len(expected_ids)
    for action_id in expected_ids:
        assert encoded["action_mask"][action_id] == 1


def test_observation_encoder_does_not_change_when_unobserved_draw_order_changes() -> None:
    encoder = ObservationEncoder(max_hand_size=4, max_players=3)
    first_engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), SkipCard()]),
            PlayerState("player_2", [DefuseCard()]),
        ],
        [NormalCard(), SeeTheFutureCard()],
    )
    second_engine = GameEngine(
        [
            PlayerState("player_1", [DefuseCard(), SkipCard()]),
            PlayerState("player_2", [DefuseCard()]),
        ],
        [SeeTheFutureCard(), NormalCard()],
    )

    np.testing.assert_array_equal(
        encoder.encode_observation(first_engine.observe("player_1")),
        encoder.encode_observation(second_engine.observe("player_1")),
    )
