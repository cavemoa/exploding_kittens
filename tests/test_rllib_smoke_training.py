from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from exploding_kittens import pettingzoo_env
from scripts.train_rllib_pettingzoo import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TRAINING_CONFIG,
    RLLibActionMaskObservationWrapper,
    apply_cli_overrides,
    balanced_evaluation_seats,
    build_ppo_config,
    build_randomized_seat_rank_log_row,
    build_training_metric_row,
    dashboard_logging_enabled_override,
    build_policy_mapping_fn,
    deep_merge_config,
    evaluate_checkpoint_by_native_seat,
    evaluate_checkpoint_by_randomized_seat_rank,
    evaluate_policy_by_randomized_seat_rank,
    evaluate_checkpoint_against_scripted_baselines,
    evaluate_training_checkpoint,
    final_evaluation_highlights,
    format_duration,
    format_evaluation_win_rates,
    format_optional_float,
    format_training_evaluation_status,
    inspect_existing_run,
    initialize_algorithm_policies,
    iteration_checkpoint_name,
    log_final_vs_first_comparison,
    log_local_training_evaluations,
    log_training_evaluations,
    load_training_config,
    load_frozen_policies,
    metric_from_result,
    normalize_dashboard_logging,
    normalize_model_config,
    normalize_policy_setup,
    normalize_reward_config,
    normalize_training_config,
    prepare_config_for_existing_run,
    parse_name_list,
    previous_wandb_run_id_from_local_wandb,
    progress_bar,
    ranked_players_for_episode,
    RandomizedSeatRankEvaluationResult,
    reveal_opponent_card_counts_override,
    save_resolved_config,
    start_wandb_run,
    shared_policy_mapping,
    should_evaluate_iteration,
    should_save_iteration_checkpoint,
    train_from_config,
    train_rllib_shared_policy,
    training_config_to_payload,
    validate_training_args,
    validate_policy_setup,
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
    assert config.model["fcnet_hiddens"] == [256, 256]
    assert config.model["fcnet_activation"] == "tanh"
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
            num_env_runners=0,
            checkpoint_interval=None,
            evaluation_interval=None,
            evaluation_episodes=1,
            evaluation_opponents=("random",),
            evaluation_scope="primary-seat",
            restore_checkpoint=None,
            reward_config=normalize_reward_config({}, players=3),
            model_config=normalize_model_config({}),
            policy_setup=normalize_policy_setup({}, players=3),
            wandb_mode="disabled",
        )

    with pytest.raises(ValueError, match="wandb_mode"):
        validate_training_args(
            iterations=1,
            players=3,
            train_batch_size=32,
            minibatch_size=16,
            rollout_fragment_length=16,
            num_env_runners=0,
            checkpoint_interval=None,
            evaluation_interval=None,
            evaluation_episodes=1,
            evaluation_opponents=("random",),
            evaluation_scope="primary-seat",
            restore_checkpoint=None,
            reward_config=normalize_reward_config({}, players=3),
            model_config=normalize_model_config({}),
            policy_setup=normalize_policy_setup({}, players=3),
            wandb_mode="quiet",
        )

    with pytest.raises(ValueError, match="Unknown evaluation opponent"):
        validate_training_args(
            iterations=1,
            players=3,
            train_batch_size=32,
            minibatch_size=16,
            rollout_fragment_length=16,
            num_env_runners=0,
            checkpoint_interval=None,
            evaluation_interval=None,
            evaluation_episodes=1,
            evaluation_opponents=("mystery",),
            evaluation_scope="primary-seat",
            restore_checkpoint=None,
            reward_config=normalize_reward_config({}, players=3),
            model_config=normalize_model_config({}),
            policy_setup=normalize_policy_setup({}, players=3),
            wandb_mode="disabled",
        )


def test_parse_name_list_ignores_empty_items() -> None:
    assert parse_name_list("normal,skip,,attack") == ("normal", "skip", "attack")


def test_load_training_config_merges_nested_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "rllib.yaml"
    config_path.write_text(
        "\n".join(
            (
                "game:",
                "  players: 4",
                "  exclude_cards:",
                "    - attack",
                "training:",
                "  iterations: 2",
                "  num_env_runners: 1",
                "reward:",
                "  profile: terminal-rank",
                "  terminal_rank_rewards: [1.0, 0.3, -0.3, -1.0]",
                "model:",
                "  hidden_layers: [512, 256]",
                "  activation: relu",
                "experiment:",
                "  run_name: yaml_run",
                "  checkpoint_interval: 5",
                "  show_progress: false",
                "dashboard_logging:",
                "  enabled: true",
                "  experiment_dir: reports/test-experiments",
                "  run_name: dashboard_yaml_run",
            )
        ),
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config["game"]["players"] == 4
    assert config["game"]["exclude_cards"] == ["attack"]
    assert config["game"]["max_turns"] == DEFAULT_TRAINING_CONFIG["game"]["max_turns"]
    assert config["training"]["iterations"] == 2
    assert config["training"]["num_env_runners"] == 1
    assert config["reward"]["profile"] == "terminal-rank"
    assert config["reward"]["terminal_rank_rewards"] == [1.0, 0.3, -0.3, -1.0]
    assert config["model"]["hidden_layers"] == [512, 256]
    assert config["model"]["activation"] == "relu"
    assert config["experiment"]["run_name"] == "yaml_run"
    assert config["experiment"]["checkpoint_interval"] == 5
    assert config["experiment"]["show_progress"] is False
    assert config["dashboard_logging"]["experiment_dir"] == "reports/test-experiments"
    assert config["dashboard_logging"]["run_name"] == "dashboard_yaml_run"
    assert config["experiment"]["evaluation_opponents"] == [
        "random",
        "safe-rule",
        "draw-only",
    ]


def test_deep_merge_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="game.not_a_key"):
        deep_merge_config(
            DEFAULT_TRAINING_CONFIG,
            {"game": {"not_a_key": 1}},
        )


def test_apply_cli_overrides_prefers_command_line_values() -> None:
    args = argparse_namespace(
        players=5,
        seed=None,
        max_turns=None,
        include_cards=None,
        exclude_cards=("normal", "skip"),
        enabled_combo_rules=None,
        reveal_opponent_card_counts=False,
        hide_opponent_card_counts=True,
        iterations=3,
        checkpoint_dir=None,
        run_name="cli_run",
        train_batch_size=None,
        minibatch_size=None,
        rollout_fragment_length=None,
        num_epochs=None,
        lr=None,
        num_env_runners=None,
        model_hidden_layers=(128, 64),
        model_activation="swish",
        checkpoint_interval=None,
        evaluation_interval=10,
        evaluation_episodes=None,
        evaluation_opponents=("random",),
        evaluation_scope="native-seat",
        restore_checkpoint=Path("models/rllib/previous/checkpoint_000010"),
        existing_run_action="continue",
        continue_mode="same-run",
        wandb_mode=None,
        wandb_project=None,
        wandb_entity=None,
        no_progress=True,
        show_ray_warnings=True,
        dashboard_logging=False,
        no_dashboard_logging=True,
        experiment_dir=Path("reports/cli-experiments"),
        dashboard_run_name="dashboard_cli_run",
    )

    config = apply_cli_overrides(
        deep_merge_config(
            DEFAULT_TRAINING_CONFIG,
            {
                "game": {"players": 3, "reveal_opponent_card_counts": True},
                "training": {"iterations": 1},
                "experiment": {"run_name": "yaml_run"},
            },
        ),
        args,
    )

    assert config["game"]["players"] == 5
    assert config["game"]["exclude_cards"] == ("normal", "skip")
    assert config["game"]["reveal_opponent_card_counts"] is False
    assert config["training"]["iterations"] == 3
    assert config["model"]["hidden_layers"] == (128, 64)
    assert config["model"]["activation"] == "swish"
    assert config["experiment"]["run_name"] == "cli_run"
    assert config["experiment"]["evaluation_interval"] == 10
    assert config["experiment"]["evaluation_opponents"] == ("random",)
    assert config["experiment"]["evaluation_scope"] == "native-seat"
    assert config["experiment"]["restore_checkpoint"] == Path(
        "models/rllib/previous/checkpoint_000010"
    )
    assert config["experiment"]["existing_run_action"] == "continue"
    assert config["experiment"]["continue_mode"] == "same-run"
    assert config["experiment"]["show_progress"] is False
    assert config["experiment"]["suppress_ray_warnings"] is False
    assert config["dashboard_logging"]["enabled"] is False
    assert config["dashboard_logging"]["experiment_dir"] == Path("reports/cli-experiments")
    assert config["dashboard_logging"]["run_name"] == "dashboard_cli_run"


def test_reveal_opponent_card_counts_override_rejects_conflict() -> None:
    with pytest.raises(ValueError, match="Use only one"):
        reveal_opponent_card_counts_override(
            argparse_namespace(
                reveal_opponent_card_counts=True,
                hide_opponent_card_counts=True,
            )
        )


def test_dashboard_logging_enabled_override_rejects_conflict() -> None:
    with pytest.raises(ValueError, match="dashboard-logging"):
        dashboard_logging_enabled_override(
            argparse_namespace(
                dashboard_logging=True,
                no_dashboard_logging=True,
            )
        )


def test_normalize_dashboard_logging_defaults_to_experiment_run_name() -> None:
    normalized = normalize_dashboard_logging(
        {"enabled": True, "experiment_dir": Path("reports/custom")},
        fallback_run_name="training_run",
    )

    assert normalized == {
        "enabled": True,
        "experiment_dir": str(Path("reports/custom")),
        "run_name": "training_run",
        "flush_each_iteration": True,
    }


def test_normalize_reward_config_defaults_and_validates_rank_weights() -> None:
    assert normalize_reward_config({}, players=4) == {
        "profile": "sparse",
        "terminal_rank_rewards": [1.0, 0.3, -0.3, -1.0],
    }
    assert normalize_reward_config(
        {
            "profile": "terminal-rank",
            "terminal_rank_rewards": [1, 0.25, -0.25, -1],
        },
        players=4,
    ) == {
        "profile": "terminal-rank",
        "terminal_rank_rewards": [1.0, 0.25, -0.25, -1.0],
    }
    with pytest.raises(ValueError, match="terminal_rank_rewards"):
        normalize_reward_config(
            {"profile": "terminal-rank", "terminal_rank_rewards": [1.0, -1.0]},
            players=4,
        )


def test_normalize_model_config_defaults_and_validates_values() -> None:
    assert normalize_model_config({}) == {
        "hidden_layers": [256, 256],
        "activation": "tanh",
    }
    assert normalize_model_config(
        {"hidden_layers": "128,64", "activation": "relu"}
    ) == {
        "hidden_layers": [128, 64],
        "activation": "relu",
    }
    with pytest.raises(ValueError, match="hidden_layers"):
        normalize_model_config({"hidden_layers": [128, 0], "activation": "relu"})
    with pytest.raises(ValueError, match="activation"):
        normalize_model_config({"hidden_layers": [128], "activation": "mystery"})


def test_prepare_config_for_existing_run_returns_config_when_run_is_new(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="new_run")

    prepared = prepare_config_for_existing_run(config, interactive=False)

    assert prepared is not None
    assert prepared["experiment"]["run_name"] == "new_run"
    assert prepared["experiment"]["restore_checkpoint"] is None


def test_prepare_config_for_existing_run_continues_to_new_run_name(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    info = create_existing_run(config)

    prepared = prepare_config_for_existing_run(
        config,
        input_fn=lambda prompt: "c",
        output_fn=lambda message: None,
        interactive=True,
    )

    assert prepared is not None
    assert prepared["experiment"]["run_name"] == "existing_run_continue_001"
    assert prepared["dashboard_logging"]["run_name"] == "existing_run_continue_001"
    assert prepared["experiment"]["restore_checkpoint"] == str(info.latest_checkpoint)
    assert prepared["continuation"] == {
        "parent_run_name": "existing_run",
        "restore_checkpoint": str(info.latest_checkpoint),
        "start_iteration": 25,
        "mode": "new-run",
        "wandb_run_id": "wandb-existing-run",
    }


def test_prepare_config_for_existing_run_continues_in_same_run_name(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    config["experiment"]["continue_mode"] = "same-run"
    info = create_existing_run(config)

    prepared = prepare_config_for_existing_run(
        config,
        input_fn=lambda prompt: "c",
        output_fn=lambda message: None,
        interactive=True,
    )

    assert prepared is not None
    assert prepared["experiment"]["run_name"] == "existing_run"
    assert prepared["dashboard_logging"]["run_name"] == "existing_run"
    assert prepared["experiment"]["restore_checkpoint"] == str(info.latest_checkpoint)
    assert prepared["continuation"] == {
        "parent_run_name": "existing_run",
        "restore_checkpoint": str(info.latest_checkpoint),
        "start_iteration": 25,
        "mode": "same-run",
        "wandb_run_id": "wandb-existing-run",
    }


def test_previous_wandb_run_id_from_local_wandb_matches_run_name(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run-20260604_200410-sm1jotie" / "files"
    run_dir.mkdir(parents=True)
    run_dir.joinpath("config.yaml").write_text(
        "\n".join(
            (
                "experiment:",
                "  value:",
                "    run_name: 4seat_64_relu_150",
            )
        ),
        encoding="utf-8",
    )

    assert previous_wandb_run_id_from_local_wandb(
        "4seat_64_relu_150",
        wandb_dir=tmp_path,
    ) == "sm1jotie"


def test_prepare_config_for_existing_run_quits_when_noninteractive(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    create_existing_run(config)
    output = []

    prepared = prepare_config_for_existing_run(
        config,
        output_fn=output.append,
        interactive=False,
    )

    assert prepared is None
    assert any("stdin is not interactive" in line for line in output)


def test_prepare_config_for_existing_run_overwrite_requires_confirmation(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    config["experiment"]["existing_run_action"] = "overwrite"
    info = create_existing_run(config)

    cancelled = prepare_config_for_existing_run(
        config,
        input_fn=lambda prompt: "nope",
        output_fn=lambda message: None,
        interactive=True,
    )

    assert cancelled is None
    assert info.checkpoint_dir.exists()
    assert info.dashboard_dir.exists()

    confirmed = prepare_config_for_existing_run(
        config,
        input_fn=lambda prompt: "OVERWRITE existing_run",
        output_fn=lambda message: None,
        interactive=True,
    )

    assert confirmed is not None
    assert not info.checkpoint_dir.exists()
    assert not info.dashboard_dir.exists()
    assert confirmed["experiment"]["run_name"] == "existing_run"


def test_inspect_existing_run_marks_changed_game_config_incompatible(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    create_existing_run(config)
    changed = deep_merge_config(config, {"game": {"players": 4}})
    changed = normalize_training_config(changed)

    info = inspect_existing_run(changed)

    assert info.config_compatibility == "incompatible"
    assert "game" in info.incompatible_fields


def test_inspect_existing_run_marks_changed_model_config_incompatible(
    tmp_path: Path,
) -> None:
    config = run_guard_config(tmp_path, run_name="existing_run")
    create_existing_run(config)
    changed = deep_merge_config(config, {"model": {"hidden_layers": [512, 256]}})
    changed = normalize_training_config(changed)

    info = inspect_existing_run(changed)

    assert info.config_compatibility == "incompatible"
    assert "model" in info.incompatible_fields


def test_save_resolved_config_round_trips_payload(tmp_path: Path) -> None:
    path = tmp_path / "resolved_config.yaml"
    payload = training_config_to_payload(DEFAULT_TRAINING_CONFIG)

    save_resolved_config(path, payload)
    loaded = load_training_config(path)

    assert loaded["game"]["players"] == DEFAULT_TRAINING_CONFIG["game"]["players"]
    assert loaded["training"]["iterations"] == DEFAULT_TRAINING_CONFIG["training"]["iterations"]


def test_train_from_config_normalizes_and_delegates(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_train_rllib_shared_policy(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.train_rllib_shared_policy",
        fake_train_rllib_shared_policy,
    )
    config = deep_merge_config(
        DEFAULT_TRAINING_CONFIG,
        {
            "game": {
                "include_cards": "normal,skip",
                "exclude_cards": ["attack"],
            },
            "experiment": {
                "checkpoint_dir": str(tmp_path),
                "checkpoint_interval": "2",
                "evaluation_opponents": "random,safe-rule",
                "show_progress": False,
                "suppress_ray_warnings": False,
            },
            "dashboard_logging": {
                "enabled": True,
                "experiment_dir": str(tmp_path / "experiments"),
                "run_name": "dashboard_test",
            },
        },
    )

    result = train_from_config(config)

    assert result is not None
    assert captured["include_cards"] == ("normal", "skip")
    assert captured["exclude_cards"] == ("attack",)
    assert captured["reward_config"] == {
        "profile": "sparse",
        "terminal_rank_rewards": [1.0, 0.0, -1.0],
    }
    assert captured["model_config"] == {
        "hidden_layers": [256, 256],
        "activation": "tanh",
    }
    assert captured["checkpoint_dir"] == tmp_path
    assert captured["restore_checkpoint"] is None
    assert captured["continuation"] == {
        "parent_run_name": None,
        "restore_checkpoint": None,
        "start_iteration": 0,
        "mode": None,
        "wandb_run_id": None,
    }
    assert captured["checkpoint_interval"] == 2
    assert captured["evaluation_opponents"] == ("random", "safe-rule")
    assert captured["evaluation_scope"] == "primary-seat"
    assert captured["show_progress"] is False
    assert captured["suppress_ray_warnings"] is False
    assert captured["dashboard_logging"] == {
        "enabled": True,
        "experiment_dir": str(tmp_path / "experiments"),
        "run_name": "dashboard_test",
        "flush_each_iteration": True,
    }
    assert captured["policy_setup"]["mode"] == "shared"


def test_progress_helpers_format_console_values() -> None:
    assert progress_bar(0.5, width=10) == "[#####-----]"
    assert progress_bar(2.0, width=4) == "[####]"
    assert format_duration(65.2) == "01:05"
    assert format_duration(3661.0) == "1:01:01"
    assert format_optional_float(None) == "n/a"
    assert format_optional_float(1.23456) == "1.235"


def test_training_evaluation_status_formats_compact_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.time.perf_counter",
        lambda: 150.0,
    )

    summary = format_training_evaluation_status(
        iteration=5,
        total_iterations=20,
        started_at=100.0,
        evaluation_results={
            "random": FakeEvaluationResult(rllib_combined_win_rate=0.25),
            "safe-rule": FakeEvaluationResult(rllib_combined_win_rate=0.5),
            "draw-only": FakeEvaluationResult(rllib_combined_win_rate=0.75),
        },
    )

    assert summary == (
        "iteration 5/20 elapsed=00:50 eta=02:30 "
        "win_rates random=0.250 safe-rule=0.500 draw-only=0.750"
    )
    assert format_evaluation_win_rates(
        {
            "random": {
                "player_1": FakeEvaluationResult(rllib_combined_win_rate=0.25),
                "player_2": FakeEvaluationResult(rllib_combined_win_rate=0.5),
            }
        }
    ) == "random: p1=0.250 p2=0.500"
    assert format_evaluation_win_rates({}) == "n/a"


def test_build_training_metric_row_tracks_dashboard_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.time.perf_counter",
        lambda: 125.0,
    )

    row = build_training_metric_row(
        iteration=2,
        total_iterations=5,
        started_at=100.0,
        result={
            "num_env_steps_sampled": 100,
            "num_agent_steps_sampled": 250,
            "episode_reward_mean": 0.25,
            "episode_len_mean": 12.0,
        },
    )

    assert row["iteration"] == 2
    assert row["total_iterations"] == 5
    assert row["elapsed_seconds"] == 25.0
    assert row["estimated_seconds_remaining"] == 37.5
    assert row["environment_steps_sampled"] == 100
    assert row["agent_steps_sampled"] == 250
    assert row["environment_steps_per_second"] == 4.0
    assert row["episode_reward_mean"] == 0.25
    assert row["episode_length_mean"] == 12.0


def test_iteration_checkpoint_helpers() -> None:
    assert iteration_checkpoint_name(12) == "checkpoint_000012"
    assert should_save_iteration_checkpoint(1, None, 10)
    assert should_save_iteration_checkpoint(10, 10, None)
    assert not should_save_iteration_checkpoint(2, 10, None)
    assert should_evaluate_iteration(20, 10)
    assert not should_evaluate_iteration(3, 10)


def test_normalize_policy_setup_keeps_shared_as_default() -> None:
    setup = normalize_policy_setup({}, players=3)

    assert setup["mode"] == "shared"
    assert setup["seat_policies"] == {
        "player_1": "shared_policy",
        "player_2": "shared_policy",
        "player_3": "shared_policy",
    }
    assert setup["trainable_policies"] == ["shared_policy"]
    validate_policy_setup(setup, players=3)


def test_normalize_policy_setup_builds_separate_per_seat_mapping() -> None:
    setup = normalize_policy_setup({"mode": "separate-per-seat"}, players=3)

    assert setup["seat_policies"] == {
        "player_1": "player_1_policy",
        "player_2": "player_2_policy",
        "player_3": "player_3_policy",
    }
    assert setup["trainable_policies"] == [
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
    ]
    validate_policy_setup(setup, players=3)


def test_four_player_separate_per_seat_longer_config_is_valid() -> None:
    config = normalize_training_config(load_training_config(
        Path("configs/rllib_separate_per_seat_4p_longer.yaml")
    ))

    assert config["game"]["players"] == 4
    assert config["reward"] == {
        "profile": "terminal-rank",
        "terminal_rank_rewards": [1.0, 0.3, -0.3, -1.0],
    }
    assert config["model"] == {
        "hidden_layers": [64, 64, 64],
        "activation": "relu",
    }
    assert config["policy_setup"]["mode"] == "separate-per-seat"
    assert config["policy_setup"]["seat_policies"] == {
        "player_1": "player_1_policy",
        "player_2": "player_2_policy",
        "player_3": "player_3_policy",
        "player_4": "player_4_policy",
    }
    assert config["policy_setup"]["trainable_policies"] == [
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
        "player_4_policy",
    ]
    assert config["experiment"]["evaluation_interval"] == 10
    assert config["experiment"]["evaluation_scope"] == "randomized-seat-rank"
    assert config["experiment"]["continue_mode"] == "same-run"
    validate_policy_setup(config["policy_setup"], players=4)


def test_normalize_policy_setup_accepts_learner_vs_frozen() -> None:
    setup = normalize_policy_setup(
        {
            "mode": "learner-vs-frozen",
            "seat_policies": {
                "player_1": "learner_policy",
                "player_2": "frozen_policy",
            },
            "trainable_policies": ["learner_policy"],
            "frozen_policies": {
                "frozen_policy": "models/rllib/smoke/checkpoint_000001",
            },
        },
        players=2,
    )

    assert setup["frozen_policies"]["frozen_policy"] == {
        "checkpoint_path": "models/rllib/smoke/checkpoint_000001",
        "policy_id": "shared_policy",
    }
    validate_policy_setup(setup, players=2)


def test_normalize_policy_setup_accepts_learner_vs_pool(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_000001"
    checkpoint.mkdir()
    metadata_path = tmp_path / "policy_pool.json"
    metadata_path.write_text(
        """
entries:
  - name: latest:checkpoint_000001
    policy_spec: rllib:{checkpoint}
    kind: rllib
    role: latest
    checkpoint_path: {checkpoint}
    iteration: 1
    score:
""".format(checkpoint=checkpoint),
        encoding="utf-8",
    )

    setup = normalize_policy_setup(
        {
            "mode": "learner-vs-pool",
            "trainable_policies": ["learner_policy"],
            "opponent_pool": {
                "metadata_path": str(metadata_path),
                "sample_mode": "latest-only",
                "seed": 1,
            },
        },
        players=3,
    )

    assert setup["seat_policies"] == {
        "player_1": "learner_policy",
        "player_2": "player_2_frozen_policy",
        "player_3": "player_3_frozen_policy",
    }
    assert setup["frozen_policies"]["player_2_frozen_policy"] == {
        "checkpoint_path": str(checkpoint),
        "policy_id": "shared_policy",
    }
    validate_policy_setup(setup, players=3)


def test_validate_policy_setup_rejects_trainable_frozen_overlap() -> None:
    setup = normalize_policy_setup(
        {
            "mode": "learner-vs-frozen",
            "seat_policies": {
                "player_1": "learner_policy",
                "player_2": "learner_policy",
            },
            "trainable_policies": ["learner_policy"],
            "frozen_policies": {
                "learner_policy": "models/rllib/smoke/checkpoint_000001",
            },
        },
        players=2,
    )

    with pytest.raises(ValueError, match="both trainable and frozen"):
        validate_policy_setup(setup, players=2)


def test_build_policy_mapping_fn_maps_seats() -> None:
    mapping_fn = build_policy_mapping_fn(
        {"player_1": "player_1_policy", "player_2": "player_2_policy"}
    )

    assert mapping_fn("player_1", object()) == "player_1_policy"
    with pytest.raises(ValueError, match="No policy configured"):
        mapping_fn("player_3", object())


def test_build_ppo_config_uses_policy_setup() -> None:
    setup = normalize_policy_setup({"mode": "separate-per-seat"}, players=3)
    config = build_ppo_config(
        env_name="test_env",
        train_batch_size=32,
        minibatch_size=16,
        rollout_fragment_length=16,
        num_epochs=1,
        lr=0.0003,
        model_config={"hidden_layers": [128, 64], "activation": "relu"},
        policy_setup=setup,
    )

    assert set(config.policies) == {
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
    }
    assert set(config.policies_to_train) == {
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
    }
    assert config.model["fcnet_hiddens"] == [128, 64]
    assert config.model["fcnet_activation"] == "relu"
    assert config.policy_mapping_fn("player_2", object()) == "player_2_policy"


def test_load_frozen_policies_sets_target_weights(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    target_policy = FakePolicy()
    source_policy = FakePolicy(weights={"w": 42})

    class FakeAlgorithm:
        def get_policy(self, policy_id):
            assert policy_id == "frozen_policy"
            return target_policy

    def fake_from_checkpoint(path, policy_ids):
        assert Path(path) == checkpoint.resolve()
        assert policy_ids == ["shared_policy"]
        return {"shared_policy": source_policy}

    from ray.rllib.policy.policy import Policy

    monkeypatch.setattr(Policy, "from_checkpoint", staticmethod(fake_from_checkpoint))
    load_frozen_policies(
        FakeAlgorithm(),
        {
            "frozen_policies": {
                "frozen_policy": {
                    "checkpoint_path": str(checkpoint),
                    "policy_id": "shared_policy",
                }
            }
        },
    )

    assert target_policy.weights == {"w": 42}


def test_initialize_algorithm_policies_restores_checkpoint(
    monkeypatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint_000100"
    checkpoint.mkdir()
    algorithm = FakeAlgorithm()

    def fake_load_frozen_policies(_algorithm, _policy_setup):
        raise AssertionError("frozen policies should not load during restore")

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.load_frozen_policies",
        fake_load_frozen_policies,
    )

    initialize_algorithm_policies(
        algorithm,
        restore_checkpoint=checkpoint,
        policy_setup={},
    )

    assert algorithm.restored_from == str(checkpoint.resolve())


def test_initialize_algorithm_policies_loads_frozen_without_restore(monkeypatch) -> None:
    calls = []

    def fake_load_frozen_policies(algorithm, policy_setup):
        calls.append((algorithm, policy_setup))

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.load_frozen_policies",
        fake_load_frozen_policies,
    )
    algorithm = FakeAlgorithm()
    policy_setup = {"frozen_policies": {}}

    initialize_algorithm_policies(
        algorithm,
        restore_checkpoint=None,
        policy_setup=policy_setup,
    )

    assert calls == [(algorithm, policy_setup)]
    assert algorithm.restored_from is None


def test_evaluate_checkpoint_against_scripted_baselines_builds_seat_policies(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = []

    def fake_evaluate_rllib_checkpoint(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(
        "scripts.evaluate_rllib_checkpoint.evaluate_rllib_checkpoint",
        fake_evaluate_rllib_checkpoint,
    )

    results = evaluate_checkpoint_against_scripted_baselines(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=3,
        players=3,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random", "safe-rule"),
    )

    assert set(results) == {"random", "safe-rule"}
    assert captured[0]["seat_policies"] == {
        "player_1": f"rllib:{tmp_path / 'checkpoint'}",
        "player_2": "random",
        "player_3": "random",
    }
    assert captured[1]["seat_policies"]["player_2"] == "safe-rule"
    assert captured[0]["seed"] == 3007
    assert captured[0]["policy_id"] == "shared_policy"


def test_evaluate_checkpoint_against_scripted_baselines_uses_primary_seat_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = []

    def fake_evaluate_rllib_checkpoint(**kwargs):
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(
        "scripts.evaluate_rllib_checkpoint.evaluate_rllib_checkpoint",
        fake_evaluate_rllib_checkpoint,
    )

    evaluate_checkpoint_against_scripted_baselines(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=25,
        players=4,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random",),
        policy_setup=normalize_policy_setup({"mode": "separate-per-seat"}, players=4),
    )

    assert captured[0]["policy_id"] == "player_1_policy"
    assert captured[0]["seat_policies"] == {
        "player_1": f"rllib:{tmp_path / 'checkpoint'}",
        "player_2": "random",
        "player_3": "random",
        "player_4": "random",
    }


def test_evaluate_checkpoint_by_native_seat_builds_each_trained_seat(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = []

    def fake_evaluate_rllib_checkpoint(**kwargs):
        captured.append(kwargs)
        return FakeEvaluationResult(rllib_combined_win_rate=0.25)

    monkeypatch.setattr(
        "scripts.evaluate_rllib_checkpoint.evaluate_rllib_checkpoint",
        fake_evaluate_rllib_checkpoint,
    )
    policy_setup = normalize_policy_setup({"mode": "separate-per-seat"}, players=4)

    results = evaluate_checkpoint_by_native_seat(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=25,
        players=4,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random",),
        policy_setup=policy_setup,
    )

    assert set(results["random"]) == {"player_1", "player_2", "player_3", "player_4"}
    assert [call["policy_id"] for call in captured] == [
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
        "player_4_policy",
    ]
    assert captured[2]["seat_policies"] == {
        "player_1": "random",
        "player_2": "random",
        "player_3": f"rllib:{tmp_path / 'checkpoint'}",
        "player_4": "random",
    }


def test_balanced_evaluation_seats_spreads_games_evenly() -> None:
    schedule = balanced_evaluation_seats(
        player_names=("player_1", "player_2", "player_3", "player_4"),
        episodes=10,
        seed=5,
    )

    assert len(schedule) == 10
    assert {seat: schedule.count(seat) for seat in set(schedule)} == {
        "player_1": 3,
        "player_2": 3,
        "player_3": 2,
        "player_4": 2,
    }


def test_ranked_players_follow_winner_then_reverse_elimination_order() -> None:
    assert ranked_players_for_episode(
        player_names=("player_1", "player_2", "player_3", "player_4"),
        winner="player_2",
        elimination_order=("player_3", "player_1", "player_4"),
    ) == ("player_2", "player_4", "player_1", "player_3")


def test_evaluate_checkpoint_by_randomized_seat_rank_scores_each_trainable_policy(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = []

    def fake_evaluate_policy_by_randomized_seat_rank(**kwargs):
        captured.append(kwargs)
        return RandomizedSeatRankEvaluationResult(
            policy_id=kwargs["policy_id"],
            episodes=kwargs["episodes"],
            wins=1,
            win_rate=0.5,
            average_rank_reward=0.25,
            average_finish_position=2.0,
            place_counts={1: 1, 2: 1, 3: 0, 4: 0},
            place_rates={1: 0.5, 2: 0.5, 3: 0.0, 4: 0.0},
            seat_counts={"player_1": 1, "player_2": 1, "player_3": 0, "player_4": 0},
            seat_win_rates={
                "player_1": 1.0,
                "player_2": 0.0,
                "player_3": 0.0,
                "player_4": 0.0,
            },
            opponent_counts={"random": 3, "safe-rule": 3},
            rank_rewards=kwargs["rank_rewards"],
        )

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.evaluate_policy_by_randomized_seat_rank",
        fake_evaluate_policy_by_randomized_seat_rank,
    )

    results = evaluate_checkpoint_by_randomized_seat_rank(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=25,
        players=4,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random", "safe-rule"),
        evaluation_rank_rewards=(1.0, 0.3, -0.3, -1.0),
        policy_setup=normalize_policy_setup({"mode": "separate-per-seat"}, players=4),
    )

    assert set(results) == {
        "player_1_policy",
        "player_2_policy",
        "player_3_policy",
        "player_4_policy",
    }
    assert captured[0]["opponent_pool"] == ("random", "safe-rule")
    assert captured[0]["rank_rewards"] == (1.0, 0.3, -0.3, -1.0)


def test_evaluate_policy_by_randomized_seat_rank_uses_balanced_rank_rewards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_load_rllib_policy(checkpoint_path, *, policy_id):
        del checkpoint_path, policy_id
        return object()

    def fake_run_multi_policy_episode(**kwargs):
        rllib_seat = next(
            seat
            for seat, controller in kwargs["controllers"].items()
            if getattr(controller, "is_trained", False)
        )
        if rllib_seat == "player_1":
            return SimpleNamespace(winner="player_1", elimination_order=())
        if rllib_seat == "player_2":
            return SimpleNamespace(
                winner="player_1",
                elimination_order=("player_4", "player_3", "player_2"),
            )
        if rllib_seat == "player_3":
            return SimpleNamespace(
                winner="player_1",
                elimination_order=("player_4", "player_3", "player_2"),
            )
        return SimpleNamespace(
            winner="player_1",
            elimination_order=("player_4", "player_3", "player_2"),
        )

    monkeypatch.setattr(
        "scripts.evaluate_rllib_checkpoint.load_rllib_policy",
        fake_load_rllib_policy,
    )
    monkeypatch.setattr(
        "scripts.evaluate_multi_policy_game.run_multi_policy_episode",
        fake_run_multi_policy_episode,
    )

    result = evaluate_policy_by_randomized_seat_rank(
        checkpoint_path=tmp_path / "checkpoint",
        policy_id="player_1_policy",
        players=4,
        seed=11,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        episodes=4,
        opponent_pool=("random", "safe-rule", "draw-only"),
        rank_rewards=(1.0, 0.3, -0.3, -1.0),
    )

    assert result.win_rate == 0.25
    assert result.average_rank_reward == pytest.approx(0.0)
    assert result.average_finish_position == 2.5
    assert result.seat_counts == {
        "player_1": 1,
        "player_2": 1,
        "player_3": 1,
        "player_4": 1,
    }
    assert result.place_counts == {1: 1, 2: 1, 3: 1, 4: 1}


def test_evaluate_training_checkpoint_dispatches_native_seat_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_evaluate_checkpoint_by_native_seat(**kwargs):
        captured.update(kwargs)
        return {"random": {"player_1": FakeEvaluationResult()}}

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.evaluate_checkpoint_by_native_seat",
        fake_evaluate_checkpoint_by_native_seat,
    )

    results = evaluate_training_checkpoint(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=1,
        players=3,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random",),
        evaluation_rank_rewards=(1.0, 0.0, -1.0),
        policy_setup=normalize_policy_setup({"mode": "separate-per-seat"}, players=3),
        evaluation_scope="native-seat",
    )

    assert set(results["random"]) == {"player_1"}
    assert captured["checkpoint_path"] == tmp_path / "checkpoint"
    assert captured["players"] == 3


def test_evaluate_training_checkpoint_dispatches_randomized_seat_rank_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured = {}

    def fake_evaluate_checkpoint_by_randomized_seat_rank(**kwargs):
        captured.update(kwargs)
        return {"player_1_policy": FakeRandomizedSeatRankResult()}

    monkeypatch.setattr(
        "scripts.train_rllib_pettingzoo.evaluate_checkpoint_by_randomized_seat_rank",
        fake_evaluate_checkpoint_by_randomized_seat_rank,
    )

    results = evaluate_training_checkpoint(
        checkpoint_path=tmp_path / "checkpoint",
        iteration=1,
        players=3,
        seed=7,
        max_turns=50,
        include_cards=None,
        exclude_cards=(),
        enabled_combo_rules=(),
        reveal_opponent_card_counts=False,
        evaluation_episodes=2,
        evaluation_opponents=("random", "safe-rule"),
        evaluation_rank_rewards=(1.0, 0.0, -1.0),
        policy_setup=normalize_policy_setup({"mode": "separate-per-seat"}, players=3),
        evaluation_scope="randomized-seat-rank",
    )

    assert set(results) == {"player_1_policy"}
    assert captured["evaluation_opponents"] == ("random", "safe-rule")
    assert captured["evaluation_rank_rewards"] == (1.0, 0.0, -1.0)


def test_log_training_evaluations_logs_diagnostics() -> None:
    run = FakeRun()
    result = FakeEvaluationResult(
        rllib_combined_win_rate=0.5,
        scripted_opponent_win_rate=0.5,
        seat_order_advantage=0.25,
        illegal_action_rate=0.0,
        combo_count=2,
        action_distribution={"draw": 3},
        card_distribution={"two_of_a_kind": 2},
    )

    log_training_evaluations(run, 4, {"random": result})

    payload, step = run.logs[0]
    assert step == 4
    assert payload["rllib_eval_during_training/random/rllib_combined_win_rate"] == 0.5
    assert payload["rllib_eval_during_training/random/illegal_action_rate"] == 0.0
    assert payload["rllib_eval_during_training/random/action_distribution/draw"] == 3
    assert payload["rllib_eval_during_training/random/card_distribution/two_of_a_kind"] == 2


def test_log_training_evaluations_logs_native_seat_wandb_metrics() -> None:
    run = FakeRun()

    log_training_evaluations(
        run,
        25,
        {
            "random": {
                "player_1": FakeEvaluationResult(rllib_combined_win_rate=0.25),
                "player_2": FakeEvaluationResult(rllib_combined_win_rate=0.5),
            }
        },
    )

    payload, step = run.logs[0]
    assert step == 25
    assert payload["eval_by_player/random/player_1/win_rate"] == 0.25
    assert payload["eval_by_player/random/player_2/win_rate"] == 0.5


def test_log_training_evaluations_logs_randomized_rank_metrics() -> None:
    run = FakeRun()

    log_training_evaluations(
        run,
        25,
        {
            "player_1_policy": FakeRandomizedSeatRankResult(
                win_rate=0.4,
                average_rank_reward=0.15,
                average_finish_position=2.2,
            )
        },
    )

    payload, step = run.logs[0]
    assert step == 25
    assert payload["general_eval/player_1_policy/win_rate"] == 0.4
    assert payload["general_eval/player_1_policy/average_rank_reward"] == 0.15
    assert payload["general_eval/player_1_policy/average_finish_position"] == 2.2
    assert payload["general_eval/player_1_policy/place_1_rate"] == 0.4
    assert payload["general_eval/player_1_policy/seat_win_rate/player_1"] == 0.25


def test_log_local_training_evaluations_writes_native_seat_rows() -> None:
    logger = FakeLocalLogger()
    policy_setup = normalize_policy_setup({"mode": "separate-per-seat"}, players=2)

    log_local_training_evaluations(
        logger,
        iteration=25,
        evaluation_episodes=2,
        evaluation_results={
            "safe-rule": {
                "player_2": FakeEvaluationResult(rllib_combined_win_rate=0.75)
            }
        },
        policy_setup=policy_setup,
    )

    assert logger.rows == [
        {
            "iteration": 25,
            "opponent_strategy": "safe-rule",
            "evaluation_episodes": 2,
            "evaluation_scope": "native-seat",
            "evaluated_seat": "player_2",
            "policy_id": "player_2_policy",
            "native_seat": True,
            "rllib_combined_win_rate": 0.75,
            "scripted_opponent_win_rate": 0.0,
            "seat_order_advantage": 0.0,
            "illegal_action_rate": 0.0,
            "combo_count": 0,
            "action_distribution": {},
            "card_distribution": {},
        }
    ]


def test_log_local_training_evaluations_writes_randomized_rank_rows() -> None:
    logger = FakeLocalLogger()
    policy_setup = normalize_policy_setup({"mode": "separate-per-seat"}, players=4)

    log_local_training_evaluations(
        logger,
        iteration=25,
        evaluation_episodes=100,
        evaluation_results={
            "player_2_policy": FakeRandomizedSeatRankResult(
                win_rate=0.35,
                average_rank_reward=0.1,
                average_finish_position=2.4,
            )
        },
        policy_setup=policy_setup,
    )

    assert logger.rows[0]["evaluation_scope"] == "randomized-seat-rank"
    assert logger.rows[0]["opponent_strategy"] == "mixed-random"
    assert logger.rows[0]["evaluated_seat"] == "balanced-random"
    assert logger.rows[0]["policy_id"] == "player_2_policy"
    assert logger.rows[0]["rllib_combined_win_rate"] == 0.35
    assert logger.rows[0]["average_rank_reward"] == 0.1
    assert logger.rows[0]["seat_counts"]["player_1"] == 25


def test_log_final_vs_first_comparison_logs_win_rate_delta() -> None:
    run = FakeRun()

    log_final_vs_first_comparison(
        run,
        final_results={"random": FakeEvaluationResult(rllib_combined_win_rate=0.75)},
        first_results={"random": FakeEvaluationResult(rllib_combined_win_rate=0.25)},
    )

    assert run.logs[0][0][
        "rllib_eval_final_vs_first/random/win_rate_delta"
    ] == 0.5


def test_log_final_vs_first_comparison_logs_randomized_rank_deltas() -> None:
    run = FakeRun()

    log_final_vs_first_comparison(
        run,
        final_results={
            "player_1_policy": FakeRandomizedSeatRankResult(
                win_rate=0.5,
                average_rank_reward=0.2,
            )
        },
        first_results={
            "player_1_policy": FakeRandomizedSeatRankResult(
                win_rate=0.25,
                average_rank_reward=-0.1,
            )
        },
    )

    payload = run.logs[0][0]
    assert payload[
        "rllib_eval_final_vs_first/general_eval/player_1_policy/win_rate_delta"
    ] == 0.25
    assert payload[
        "rllib_eval_final_vs_first/general_eval/player_1_policy/rank_reward_delta"
    ] == pytest.approx(0.3)


def test_start_wandb_run_uses_resume_id(monkeypatch) -> None:
    captured = {}

    class FakeWandb:
        @staticmethod
        def init(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(id=kwargs.get("id"))

    monkeypatch.setitem(sys.modules, "wandb", FakeWandb)

    run = start_wandb_run(
        mode="online",
        project="exploding-kittens",
        entity=None,
        name="existing_run",
        config={},
        resume_id="wandb-existing-run",
        resume="allow",
    )

    assert run.id == "wandb-existing-run"
    assert captured["id"] == "wandb-existing-run"
    assert captured["resume"] == "allow"
    assert captured["name"] == "existing_run"


def test_final_evaluation_highlights_summarizes_dashboard_fields() -> None:
    highlights = final_evaluation_highlights(
        {
            "random": FakeEvaluationResult(
                rllib_combined_win_rate=0.25,
                scripted_opponent_win_rate=0.75,
                illegal_action_rate=0.0,
                combo_count=3,
            )
        }
    )

    assert highlights == {
        "random": {
            "rllib_combined_win_rate": 0.25,
            "scripted_opponent_win_rate": 0.75,
            "illegal_action_rate": 0.0,
            "combo_count": 3,
        }
    }

    randomized = final_evaluation_highlights(
        {
            "player_1_policy": FakeRandomizedSeatRankResult(
                win_rate=0.45,
                average_rank_reward=0.2,
                average_finish_position=2.1,
            )
        }
    )

    assert randomized["player_1_policy"]["win_rate"] == 0.45
    assert randomized["player_1_policy"]["average_rank_reward"] == 0.2


def test_tiny_rllib_training_writes_dashboard_logs(tmp_path: Path) -> None:
    result = train_rllib_shared_policy(
        iterations=1,
        players=3,
        seed=23,
        max_turns=30,
        checkpoint_dir=tmp_path / "models",
        run_name="dashboard_smoke",
        train_batch_size=16,
        minibatch_size=8,
        rollout_fragment_length=8,
        num_epochs=1,
        checkpoint_interval=None,
        evaluation_interval=None,
        wandb_mode="disabled",
        show_progress=False,
        dashboard_logging={
            "enabled": True,
            "experiment_dir": str(tmp_path / "experiments"),
            "run_name": "dashboard_smoke",
            "flush_each_iteration": True,
        },
    )

    run_dir = tmp_path / "experiments" / "dashboard_smoke"

    assert result.iterations == 1
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoints.jsonl").exists()
    assert (run_dir / "summary.json").exists()
    assert '"status": "completed"' in (run_dir / "summary.json").read_text(
        encoding="utf-8"
    )
    assert '"iteration": 1' in (run_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    )


def argparse_namespace(**overrides):
    defaults = {
        "iterations": None,
        "players": None,
        "seed": None,
        "max_turns": None,
        "include_cards": None,
        "exclude_cards": None,
        "enabled_combo_rules": None,
        "reveal_opponent_card_counts": False,
        "hide_opponent_card_counts": False,
        "checkpoint_dir": None,
        "run_name": None,
        "train_batch_size": None,
        "minibatch_size": None,
        "rollout_fragment_length": None,
        "num_epochs": None,
        "lr": None,
        "num_env_runners": None,
        "model_hidden_layers": None,
        "model_activation": None,
        "checkpoint_interval": None,
        "evaluation_interval": None,
        "evaluation_episodes": None,
        "evaluation_opponents": None,
        "evaluation_scope": None,
        "restore_checkpoint": None,
        "existing_run_action": None,
        "continue_mode": None,
        "wandb_mode": None,
        "wandb_project": None,
        "wandb_entity": None,
        "no_progress": False,
        "show_ray_warnings": False,
        "dashboard_logging": False,
        "no_dashboard_logging": False,
        "experiment_dir": None,
        "dashboard_run_name": None,
    }
    defaults.update(overrides)
    return type("Args", (), defaults)()


def run_guard_config(tmp_path: Path, *, run_name: str):
    return deep_merge_config(
        DEFAULT_TRAINING_CONFIG,
        {
            "experiment": {
                "checkpoint_dir": str(tmp_path / "models"),
                "run_name": run_name,
                "wandb_mode": "disabled",
            },
            "dashboard_logging": {
                "enabled": True,
                "experiment_dir": str(tmp_path / "experiments"),
                "run_name": run_name,
            },
        },
    )


def create_existing_run(config: dict) -> object:
    normalized = normalize_training_config(config)
    run_name = normalized["experiment"]["run_name"]
    checkpoint_dir = Path(normalized["experiment"]["checkpoint_dir"]) / run_name
    dashboard_dir = (
        Path(normalized["dashboard_logging"]["experiment_dir"])
        / normalized["dashboard_logging"]["run_name"]
    )
    (checkpoint_dir / "checkpoint_000010").mkdir(parents=True)
    (checkpoint_dir / "checkpoint_000025").mkdir(parents=True)
    dashboard_dir.mkdir(parents=True)
    (dashboard_dir / "summary.json").write_text(
        '{"status": "completed", "wandb_run_id": "wandb-existing-run"}',
        encoding="utf-8",
    )
    save_resolved_config(dashboard_dir / "resolved_config.yaml", normalized)
    return inspect_existing_run(normalized)


class FakeRun:
    def __init__(self) -> None:
        self.logs = []

    def log(self, data, step=None) -> None:
        self.logs.append((data, step))


class FakeLocalLogger:
    def __init__(self) -> None:
        self.rows = []

    def log_evaluation(self, row) -> None:
        self.rows.append(row)


class FakeAlgorithm:
    def __init__(self) -> None:
        self.restored_from = None

    def restore(self, path: str) -> None:
        self.restored_from = path


class FakeEvaluationResult:
    def __init__(
        self,
        *,
        rllib_combined_win_rate=0.0,
        scripted_opponent_win_rate=0.0,
        seat_order_advantage=0.0,
        illegal_action_rate=0.0,
        combo_count=0,
        action_distribution=None,
        card_distribution=None,
    ) -> None:
        self.rllib_combined_win_rate = rllib_combined_win_rate
        self.scripted_opponent_win_rate = scripted_opponent_win_rate
        self.seat_order_advantage = seat_order_advantage
        self.illegal_action_rate = illegal_action_rate
        self.combo_count = combo_count
        self.action_distribution = action_distribution or {}
        self.card_distribution = card_distribution or {}


class FakeRandomizedSeatRankResult:
    def __init__(
        self,
        *,
        win_rate=0.4,
        average_rank_reward=0.1,
        average_finish_position=2.5,
    ) -> None:
        self.policy_id = "player_1_policy"
        self.episodes = 100
        self.wins = int(round(win_rate * self.episodes))
        self.win_rate = win_rate
        self.average_rank_reward = average_rank_reward
        self.average_finish_position = average_finish_position
        self.place_counts = {1: self.wins, 2: 20, 3: 20, 4: 100 - self.wins - 40}
        self.place_rates = {
            place: count / self.episodes for place, count in self.place_counts.items()
        }
        self.seat_counts = {
            "player_1": 25,
            "player_2": 25,
            "player_3": 25,
            "player_4": 25,
        }
        self.seat_win_rates = {
            "player_1": 0.25,
            "player_2": 0.30,
            "player_3": 0.35,
            "player_4": 0.40,
        }
        self.opponent_counts = {"random": 100, "safe-rule": 100, "draw-only": 100}
        self.rank_rewards = (1.0, 0.3, -0.3, -1.0)


class FakePolicy:
    def __init__(self, weights=None) -> None:
        self.weights = weights or {}

    def get_weights(self):
        return self.weights

    def set_weights(self, weights) -> None:
        self.weights = weights
