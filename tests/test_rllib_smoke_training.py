from pathlib import Path

import pytest

from exploding_kittens import pettingzoo_env
from scripts.train_rllib_pettingzoo import (
    DEFAULT_MODEL_NAME,
    RLLibActionMaskObservationWrapper,
    build_ppo_config,
    metric_from_result,
    parse_name_list,
    shared_policy_mapping,
    validate_training_args,
)


pytest.importorskip("ray")


def test_rllib_observation_wrapper_renames_observation_key() -> None:
    environment = RLLibActionMaskObservationWrapper(pettingzoo_env(players=3))
    environment.reset(seed=1)

    observation = environment.observe("player_1")
    observation_space = environment.observation_space("player_1")

    assert set(observation) == {"observations", "action_mask"}
    assert set(observation_space.spaces) == {"observations", "action_mask"}
    assert observation_space.contains(observation)


def test_shared_policy_mapping_returns_shared_policy() -> None:
    assert shared_policy_mapping("player_1", object()) == "shared_policy"


def test_metric_from_result_reads_top_level_and_env_runner_metrics() -> None:
    assert metric_from_result({"episode_reward_mean": 1.5}, "episode_reward_mean") == 1.5
    assert (
        metric_from_result(
            {"env_runners": {"episode_len_mean": 12.0}},
            "episode_len_mean",
        )
        == 12.0
    )
    assert metric_from_result({"env_runners": {"episode_len_mean": float("nan")}}, "episode_len_mean") is None


def test_build_ppo_config_sets_shared_policy_and_mask_model() -> None:
    config = build_ppo_config(
        env_name="test_env",
        train_batch_size=32,
        minibatch_size=16,
        rollout_fragment_length=16,
        num_epochs=1,
        lr=0.0003,
    )

    assert config.env == "test_env"
    assert config.model["custom_model"] == DEFAULT_MODEL_NAME
    assert set(config.policies) == {"shared_policy"}
    assert config.policy_mapping_fn("player_1", object()) == "shared_policy"
    assert not config.enable_rl_module_and_learner
    assert not config.enable_env_runner_and_connector_v2


def test_validate_training_args_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="minibatch_size"):
        validate_training_args(
            iterations=1,
            players=3,
            train_batch_size=16,
            minibatch_size=32,
            rollout_fragment_length=16,
            wandb_mode="disabled",
        )

    with pytest.raises(ValueError, match="wandb_mode"):
        validate_training_args(
            iterations=1,
            players=3,
            train_batch_size=32,
            minibatch_size=16,
            rollout_fragment_length=16,
            wandb_mode="quiet",
        )


def test_parse_name_list_ignores_empty_items() -> None:
    assert parse_name_list("normal,skip,,attack") == ("normal", "skip", "attack")
