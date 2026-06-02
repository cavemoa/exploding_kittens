import pytest

from scripts.train_single_agent import (
    make_model,
    make_training_env,
    validate_training_args,
)


pytest.importorskip("sb3_contrib")


def test_training_env_supports_sb3_contrib_action_masks() -> None:
    from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported

    environment = make_training_env(
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=1,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
    )

    environment.reset(seed=1)
    assert is_masking_supported(environment)
    assert get_action_masks(environment).dtype == bool
    assert get_action_masks(environment).sum() > 0
    environment.close()


def test_make_model_builds_maskable_ppo_with_multi_input_policy() -> None:
    environment = make_training_env(
        players=2,
        learner="player_1",
        opponent_strategy="draw-only",
        seed=1,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
    )

    model = make_model(
        environment,
        seed=1,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        learning_rate=0.0003,
    )

    assert type(model).__name__ == "MaskablePPO"
    assert type(model.policy).__name__ == "MaskableMultiInputActorCriticPolicy"
    environment.close()


def test_validate_training_args_rejects_bad_tiny_run_settings() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        validate_training_args(
            total_timesteps=16,
            n_steps=8,
            batch_size=16,
            n_epochs=1,
            eval_episodes=1,
            wandb_mode="disabled",
        )

    with pytest.raises(ValueError, match="wandb_mode"):
        validate_training_args(
            total_timesteps=16,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            eval_episodes=1,
            wandb_mode="quiet",
        )
