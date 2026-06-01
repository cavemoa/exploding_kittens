import numpy as np
import pytest

from exploding_kittens import Action, ActionEncoder, GameAction, GameObservation


def observation_with(
    legal_actions: tuple[GameAction, ...],
    *,
    own_hand: tuple[str, ...] = (
        "skip",
        "attack",
        "shuffle",
        "favor",
        "normal",
        "normal",
        "alter_the_future",
        "see_the_future",
    ),
) -> GameObservation:
    return GameObservation(
        player="player_1",
        own_hand=own_hand,
        hand_counts={},
        discard_counts={},
        draw_pile_size=10,
        living_players=3,
        current_turns_remaining=1,
        next_player_turns=1,
        known_top_cards=(),
        opponent_card_counts={},
        public_events=(),
        legal_actions=legal_actions,
    )


def test_action_encoder_round_trips_every_current_legal_action() -> None:
    legal_actions = (
        GameAction(kind=Action.DRAW),
        GameAction(kind=Action.PLAY_SKIP, card_indexes=(0,)),
        GameAction(kind=Action.PLAY_ATTACK, card_indexes=(1,)),
        GameAction(kind=Action.PLAY_SHUFFLE, card_indexes=(2,)),
        GameAction(kind=Action.PLAY_FAVOR, card_indexes=(3,), target_player="player_2"),
        GameAction(kind=Action.PLAY_ALTER_THE_FUTURE, card_indexes=(6,)),
        GameAction(kind=Action.PLAY_SEE_THE_FUTURE, card_indexes=(7,)),
        GameAction(
            kind=Action.PLAY_TWO_OF_A_KIND,
            card_indexes=(4, 5),
            target_player="player_3",
        ),
    )
    observation = observation_with(legal_actions)
    encoder = ActionEncoder(max_hand_size=8, max_players=3)

    action_ids = [encoder.encode(action, observation) for action in legal_actions]

    assert len(action_ids) == len(set(action_ids))
    assert [
        encoder.decode(action_id, observation) for action_id in action_ids
    ] == list(legal_actions)


def test_action_mask_marks_only_legal_action_ids() -> None:
    draw = GameAction(kind=Action.DRAW)
    skip = GameAction(kind=Action.PLAY_SKIP, card_indexes=(0,))
    combo = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(4, 5),
        target_player="player_3",
    )
    observation = observation_with((draw, skip, combo))
    encoder = ActionEncoder(max_hand_size=8, max_players=3)

    mask = encoder.action_mask(observation)
    legal_ids = {encoder.encode(action, observation) for action in observation.legal_actions}

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == np.int8
    assert mask.shape == (encoder.action_space_size,)
    assert mask.sum() == len(legal_ids)
    for action_id in legal_ids:
        assert mask[action_id] == 1
    assert mask[encoder.encode(skip, observation) + 1] == 0


def test_targeted_actions_preserve_selected_target_player() -> None:
    favor_player_2 = GameAction(
        kind=Action.PLAY_FAVOR,
        card_indexes=(3,),
        target_player="player_2",
    )
    favor_player_3 = GameAction(
        kind=Action.PLAY_FAVOR,
        card_indexes=(3,),
        target_player="player_3",
    )
    observation = observation_with((favor_player_2, favor_player_3))
    encoder = ActionEncoder(max_hand_size=8, max_players=3)

    player_2_id = encoder.encode(favor_player_2, observation)
    player_3_id = encoder.encode(favor_player_3, observation)

    assert player_2_id != player_3_id
    assert encoder.decode(player_2_id, observation).target_player == "player_2"
    assert encoder.decode(player_3_id, observation).target_player == "player_3"


def test_combo_actions_preserve_selected_card_indexes() -> None:
    normal_pair = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(4, 5),
        target_player="player_2",
    )
    skip_pair = GameAction(
        kind=Action.PLAY_TWO_OF_A_KIND,
        card_indexes=(0, 1),
        target_player="player_2",
    )
    observation = observation_with((normal_pair, skip_pair))
    encoder = ActionEncoder(max_hand_size=8, max_players=3)

    normal_pair_id = encoder.encode(normal_pair, observation)
    skip_pair_id = encoder.encode(skip_pair, observation)

    assert normal_pair_id != skip_pair_id
    assert encoder.decode(normal_pair_id, observation).card_indexes == (4, 5)
    assert encoder.decode(skip_pair_id, observation).card_indexes == (0, 1)


def test_illegal_action_ids_are_masked_out_and_do_not_decode() -> None:
    observation = observation_with((GameAction(kind=Action.DRAW),))
    encoder = ActionEncoder(max_hand_size=8, max_players=3)
    illegal_action_id = encoder.single_card_start

    assert encoder.action_mask(observation)[illegal_action_id] == 0
    with pytest.raises(ValueError, match="not legal"):
        encoder.decode(illegal_action_id, observation)


def test_encoder_reserves_tail_space_for_future_named_card_actions() -> None:
    encoder = ActionEncoder(max_hand_size=8, max_players=3, reserved_named_card_actions=12)

    assert encoder.reserved_named_card_start < encoder.action_space_size
    assert encoder.action_space_size - encoder.reserved_named_card_start == 12
