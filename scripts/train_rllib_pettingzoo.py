from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import gymnasium as gym
from pettingzoo.utils import BaseWrapper
import yaml

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only without project deps installed.
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploding_kittens import pettingzoo_env
from exploding_kittens.experiment_logging import LocalExperimentLogger


WANDB_MODES = ("online", "offline", "disabled")
REWARD_PROFILES = ("sparse", "terminal-rank")
EVALUATION_SCOPES = ("primary-seat", "native-seat", "randomized-seat-rank")
EXISTING_RUN_ACTIONS = ("ask", "continue", "overwrite", "quit")
CONTINUE_MODES = ("new-run", "same-run")
MODEL_ACTIVATIONS = ("relu", "tanh", "swish", "linear", "elu")
PRIMARY_SEAT_EVALUATION = "primary-seat"
NATIVE_SEAT_EVALUATION = "native-seat"
RANDOMIZED_SEAT_RANK_EVALUATION = "randomized-seat-rank"
DEFAULT_ENV_NAME = "exploding_kittens_rllib"
DEFAULT_MODEL_NAME = "exploding_kittens_action_mask_model"
SHARED_POLICY_ID = "shared_policy"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
DEFAULT_TRAINING_CONFIG: dict[str, Any] = {
    "game": {
        "players": 3,
        "seed": 1,
        "max_turns": 500,
        "include_cards": None,
        "exclude_cards": [],
        "enabled_combo_rules": [],
        "reveal_opponent_card_counts": False,
    },
    "reward": {
        "profile": "sparse",
        "terminal_rank_rewards": None,
    },
    "training": {
        "iterations": 1,
        "train_batch_size": 64,
        "minibatch_size": 32,
        "rollout_fragment_length": 32,
        "num_epochs": 1,
        "lr": 0.0003,
        "num_env_runners": 0,
    },
    "model": {
        "hidden_layers": [256, 256],
        "activation": "tanh",
    },
    "experiment": {
        "checkpoint_dir": "models/rllib",
        "run_name": "rllib_shared_policy_smoke",
        "checkpoint_interval": None,
        "evaluation_interval": None,
        "evaluation_episodes": 5,
        "evaluation_opponents": ["random", "safe-rule", "draw-only"],
        "evaluation_scope": PRIMARY_SEAT_EVALUATION,
        "restore_checkpoint": None,
        "existing_run_action": "ask",
        "continue_mode": "new-run",
        "wandb_mode": "disabled",
        "wandb_project": "exploding-kittens",
        "wandb_entity": None,
        "show_progress": True,
        "suppress_ray_warnings": True,
    },
    "dashboard_logging": {
        "enabled": True,
        "experiment_dir": "reports/experiments",
        "run_name": None,
        "flush_each_iteration": True,
    },
    "continuation": {
        "parent_run_name": None,
        "restore_checkpoint": None,
        "start_iteration": 0,
        "mode": None,
        "wandb_run_id": None,
    },
    "policy_setup": {
        "mode": "shared",
        "shared_policy_id": SHARED_POLICY_ID,
        "seat_policies": {},
        "trainable_policies": [],
        "frozen_policies": {},
        "opponent_pool": {
            "metadata_path": None,
            "sample_mode": "random-historical",
            "seed": 1,
        },
    },
}


@dataclass(frozen=True)
class RLLibTrainingResult:
    iterations: int
    checkpoint_path: Path
    num_env_steps_sampled: int
    num_agent_steps_sampled: int
    episode_reward_mean: float | None
    episode_len_mean: float | None
    checkpoint_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RandomizedSeatRankEvaluationResult:
    policy_id: str
    episodes: int
    wins: int
    win_rate: float
    average_rank_reward: float
    average_finish_position: float
    place_counts: dict[int, int]
    place_rates: dict[int, float]
    seat_counts: dict[str, int]
    seat_win_rates: dict[str, float]
    opponent_counts: dict[str, int]
    rank_rewards: tuple[float, ...]


@dataclass(frozen=True)
class ExistingRunInfo:
    run_name: str
    checkpoint_dir: Path
    dashboard_dir: Path
    exists: bool
    latest_checkpoint: Path | None
    previous_status: str
    config_compatibility: str
    incompatible_fields: tuple[str, ...] = ()
    wandb_run_id: str | None = None


class RLLibActionMaskObservationWrapper(BaseWrapper):
    """Rename `observation` to `observations` for RLlib's action-mask model."""

    def observation_space(self, agent: str) -> gym.Space:
        space = self.env.observation_space(agent)
        return gym.spaces.Dict(
            {
                "observations": space["observation"],
                "action_mask": space["action_mask"],
            }
        )

    def observe(self, agent: str) -> dict[str, Any] | None:
        observation = self.env.observe(agent)
        if observation is None:
            return None
        return {
            "observations": observation["observation"],
            "action_mask": observation["action_mask"],
        }


def train_rllib_shared_policy(
    *,
    iterations: int = 1,
    players: int = 3,
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    reward_config: dict[str, Any] | None = None,
    checkpoint_dir: Path = Path("models/rllib"),
    run_name: str = "rllib_shared_policy_smoke",
    train_batch_size: int = 64,
    minibatch_size: int = 32,
    rollout_fragment_length: int = 32,
    num_epochs: int = 1,
    lr: float = 0.0003,
    num_env_runners: int = 0,
    model_config: dict[str, Any] | None = None,
    checkpoint_interval: int | None = None,
    evaluation_interval: int | None = None,
    evaluation_episodes: int = 5,
    evaluation_opponents: tuple[str, ...] = ("random", "safe-rule", "draw-only"),
    evaluation_scope: str = PRIMARY_SEAT_EVALUATION,
    restore_checkpoint: Path | None = None,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
    show_progress: bool = True,
    suppress_ray_warnings: bool = True,
    dashboard_logging: dict[str, Any] | None = None,
    continuation: dict[str, Any] | None = None,
    policy_setup: dict[str, Any] | None = None,
) -> RLLibTrainingResult:
    policy_setup = normalize_policy_setup(policy_setup or {}, players=players)
    reward_config = normalize_reward_config(reward_config or {}, players=players)
    model_config = normalize_model_config(model_config or {})
    dashboard_logging = normalize_dashboard_logging(
        dashboard_logging or {},
        fallback_run_name=run_name,
    )
    validate_training_args(
        iterations=iterations,
        players=players,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        rollout_fragment_length=rollout_fragment_length,
        num_env_runners=num_env_runners,
        checkpoint_interval=checkpoint_interval,
        evaluation_interval=evaluation_interval,
        evaluation_episodes=evaluation_episodes,
        evaluation_opponents=evaluation_opponents,
        evaluation_scope=evaluation_scope,
        restore_checkpoint=restore_checkpoint,
        reward_config=reward_config,
        model_config=model_config,
        policy_setup=policy_setup,
        wandb_mode=wandb_mode,
    )
    config_payload = training_config_to_payload(
        build_training_config(
            iterations=iterations,
            players=players,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            reward_config=reward_config,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            rollout_fragment_length=rollout_fragment_length,
            num_epochs=num_epochs,
            lr=lr,
            num_env_runners=num_env_runners,
            model_config=model_config,
            checkpoint_interval=checkpoint_interval,
            evaluation_interval=evaluation_interval,
            evaluation_episodes=evaluation_episodes,
            evaluation_opponents=evaluation_opponents,
            evaluation_scope=evaluation_scope,
            restore_checkpoint=restore_checkpoint,
            wandb_mode=wandb_mode,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            show_progress=show_progress,
            suppress_ray_warnings=suppress_ray_warnings,
            dashboard_logging=dashboard_logging,
            continuation=continuation or DEFAULT_TRAINING_CONFIG["continuation"],
            policy_setup=policy_setup,
        )
    )
    start_iteration = int((continuation or {}).get("start_iteration") or 0)
    same_run_continuation = (continuation or {}).get("mode") == "same-run"
    local_logger = make_local_experiment_logger(
        dashboard_logging,
        config=config_payload,
        append_existing=same_run_continuation,
    )
    local_logger.start()
    run = None
    algorithm = None
    progress: ConsoleTrainingProgress | None = None
    ray_started_here = False

    try:
        run = start_wandb_run(
            mode=wandb_mode,
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config=config_payload,
            resume_id=(continuation or {}).get("wandb_run_id"),
            resume="allow" if same_run_continuation else None,
        )
        local_logger.set_wandb_run_id(getattr(run, "id", None))

        configure_rllib_console_noise(suppress_ray_warnings)
        progress = ConsoleTrainingProgress(
            enabled=show_progress,
            total_iterations=iterations,
            run_name=run_name,
        )
        progress.start(
            players=players,
            train_batch_size=train_batch_size,
            evaluation_interval=evaluation_interval,
        )
        training_started_at = time.perf_counter()
        ray = import_ray()
        ray_started_here = not ray.is_initialized()
        if ray_started_here:
            ray_init_kwargs: dict[str, Any] = {
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "num_cpus": 1,
            }
            if suppress_ray_warnings:
                ray_init_kwargs["logging_level"] = logging.ERROR
            ray.init(**ray_init_kwargs)

        env_name = f"{DEFAULT_ENV_NAME}_{run_name}_{seed}".replace("-", "_")
        register_rllib_env(
            env_name=env_name,
            players=players,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            reward_config=reward_config,
        )
        register_action_mask_model(DEFAULT_MODEL_NAME)
        config = build_ppo_config(
            env_name=env_name,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            rollout_fragment_length=rollout_fragment_length,
            num_epochs=num_epochs,
            lr=lr,
            num_env_runners=num_env_runners,
            model_name=DEFAULT_MODEL_NAME,
            model_config=model_config,
            policy_setup=policy_setup,
        )
        algorithm = config.build_algo()
        initialize_algorithm_policies(
            algorithm,
            restore_checkpoint=restore_checkpoint,
            policy_setup=policy_setup,
        )
        last_result: dict[str, Any] = {}
        checkpoint_root = checkpoint_dir / run_name
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_paths: list[Path] = []
        first_checkpoint_path: Path | None = (
            Path(continuation["restore_checkpoint"])
            if same_run_continuation and continuation.get("restore_checkpoint")
            else None
        )
        for iteration in range(1, iterations + 1):
            global_iteration = start_iteration + iteration
            progress.set_phase("training")
            last_result = algorithm.train()
            log_iteration_result(run, global_iteration, last_result)
            local_logger.log_metric(
                build_training_metric_row(
                    iteration=global_iteration,
                    total_iterations=start_iteration + iterations,
                    started_at=training_started_at,
                    result=last_result,
                )
            )
            progress.update(iteration, last_result)
            checkpoint_path_for_iteration: Path | None = None
            if should_save_iteration_checkpoint(
                iteration,
                checkpoint_interval,
                evaluation_interval,
            ):
                checkpoint_path_for_iteration = save_algorithm_checkpoint(
                    algorithm,
                    checkpoint_root / iteration_checkpoint_name(global_iteration),
                    config_payload,
                )
                checkpoint_paths.append(checkpoint_path_for_iteration)
                local_logger.log_checkpoint(
                    iteration=global_iteration,
                    checkpoint_path=checkpoint_path_for_iteration,
                )
                if first_checkpoint_path is None:
                    first_checkpoint_path = checkpoint_path_for_iteration
            if should_evaluate_iteration(iteration, evaluation_interval):
                if checkpoint_path_for_iteration is None:
                    checkpoint_path_for_iteration = save_algorithm_checkpoint(
                        algorithm,
                        checkpoint_root / iteration_checkpoint_name(global_iteration),
                        config_payload,
                    )
                    checkpoint_paths.append(checkpoint_path_for_iteration)
                    local_logger.log_checkpoint(
                        iteration=global_iteration,
                        checkpoint_path=checkpoint_path_for_iteration,
                    )
                    if first_checkpoint_path is None:
                        first_checkpoint_path = checkpoint_path_for_iteration
                progress.set_phase(f"evaluating {iteration}/{iterations}")
                evaluation_results = evaluate_training_checkpoint(
                    checkpoint_path=checkpoint_path_for_iteration,
                    iteration=global_iteration,
                    players=players,
                    seed=seed,
                    max_turns=max_turns,
                    include_cards=include_cards,
                    exclude_cards=exclude_cards,
                    enabled_combo_rules=enabled_combo_rules,
                    reveal_opponent_card_counts=reveal_opponent_card_counts,
                    evaluation_episodes=evaluation_episodes,
                    evaluation_opponents=evaluation_opponents,
                    evaluation_rank_rewards=tuple(
                        reward_config["terminal_rank_rewards"]
                    ),
                    policy_setup=policy_setup,
                    evaluation_scope=evaluation_scope,
                )
                log_training_evaluations(run, global_iteration, evaluation_results)
                log_local_training_evaluations(
                    local_logger,
                    iteration=global_iteration,
                    evaluation_episodes=evaluation_episodes,
                    evaluation_results=evaluation_results,
                    policy_setup=policy_setup,
                )
                progress.evaluation_summary(iteration, evaluation_results)

        checkpoint_path = checkpoint_root.resolve()
        progress.set_phase("saving final checkpoint")
        final_checkpoint_path = save_algorithm_checkpoint(
            algorithm,
            checkpoint_path,
            config_payload,
        )
        if final_checkpoint_path not in checkpoint_paths:
            checkpoint_paths.append(final_checkpoint_path)
            local_logger.log_checkpoint(
                iteration=start_iteration + iterations,
                checkpoint_path=final_checkpoint_path,
        )
        final_evaluation_results: dict[str, Any] = {}
        if evaluation_interval is not None:
            progress.set_phase("evaluating final checkpoint")
            final_evaluation_results = evaluate_training_checkpoint(
                checkpoint_path=final_checkpoint_path,
                iteration=start_iteration + iterations,
                players=players,
                seed=seed,
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                evaluation_episodes=evaluation_episodes,
                evaluation_opponents=evaluation_opponents,
                evaluation_rank_rewards=tuple(reward_config["terminal_rank_rewards"]),
                policy_setup=policy_setup,
                evaluation_scope=evaluation_scope,
            )
            log_training_evaluations(
                run,
                start_iteration + iterations,
                final_evaluation_results,
            )
            log_local_training_evaluations(
                local_logger,
                iteration=start_iteration + iterations,
                evaluation_episodes=evaluation_episodes,
                evaluation_results=final_evaluation_results,
                policy_setup=policy_setup,
            )
            progress.evaluation_summary(iterations, final_evaluation_results)
            if first_checkpoint_path is not None:
                progress.set_phase("comparing final vs first")
                first_evaluation_results = evaluate_training_checkpoint(
                    checkpoint_path=first_checkpoint_path,
                    iteration=start_iteration,
                    players=players,
                    seed=seed,
                    max_turns=max_turns,
                    include_cards=include_cards,
                    exclude_cards=exclude_cards,
                    enabled_combo_rules=enabled_combo_rules,
                    reveal_opponent_card_counts=reveal_opponent_card_counts,
                    evaluation_episodes=evaluation_episodes,
                    evaluation_opponents=evaluation_opponents,
                    evaluation_rank_rewards=tuple(
                        reward_config["terminal_rank_rewards"]
                    ),
                    policy_setup=policy_setup,
                    evaluation_scope=evaluation_scope,
                )
                log_final_vs_first_comparison(
                    run,
                    final_results=final_evaluation_results,
                    first_results=first_evaluation_results,
                )
        result = RLLibTrainingResult(
            iterations=iterations,
            checkpoint_path=checkpoint_path,
            num_env_steps_sampled=int(last_result.get("num_env_steps_sampled", 0)),
            num_agent_steps_sampled=int(last_result.get("num_agent_steps_sampled", 0)),
            episode_reward_mean=metric_from_result(last_result, "episode_reward_mean"),
            episode_len_mean=metric_from_result(last_result, "episode_len_mean"),
            checkpoint_paths=tuple(checkpoint_paths),
        )
        log_summary(run, result)
        local_logger.complete(
            final_checkpoint_path=final_checkpoint_path,
            final_evaluation_highlights=final_evaluation_highlights(
                final_evaluation_results
            ),
        )
        progress.finish(result)
        return result
    except Exception as exc:
        local_logger.fail(exc)
        raise
    finally:
        if algorithm is not None:
            algorithm.stop()
        if progress is not None:
            progress.close()
        if ray_started_here:
            ray.shutdown()
        if run is not None:
            run.finish()


def validate_training_args(
    *,
    iterations: int,
    players: int,
    train_batch_size: int,
    minibatch_size: int,
    rollout_fragment_length: int,
    num_env_runners: int,
    checkpoint_interval: int | None,
    evaluation_interval: int | None,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    evaluation_scope: str,
    restore_checkpoint: Path | None,
    reward_config: dict[str, Any],
    model_config: dict[str, Any],
    policy_setup: dict[str, Any],
    wandb_mode: str,
) -> None:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    if players < 2:
        raise ValueError("players must be at least 2.")
    if train_batch_size < 1:
        raise ValueError("train_batch_size must be at least 1.")
    if minibatch_size < 1:
        raise ValueError("minibatch_size must be at least 1.")
    if minibatch_size > train_batch_size:
        raise ValueError("minibatch_size cannot exceed train_batch_size.")
    if rollout_fragment_length < 1:
        raise ValueError("rollout_fragment_length must be at least 1.")
    if num_env_runners < 0:
        raise ValueError("num_env_runners cannot be negative.")
    if checkpoint_interval is not None and checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be at least 1 when set.")
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be at least 1 when set.")
    if evaluation_episodes < 1:
        raise ValueError("evaluation_episodes must be at least 1.")
    known_opponents = {"random", "safe-rule", "draw-only", "skip-if-possible"}
    unknown_opponents = sorted(set(evaluation_opponents) - known_opponents)
    if unknown_opponents:
        raise ValueError(
            "Unknown evaluation opponent(s): " + ", ".join(unknown_opponents)
        )
    if evaluation_scope not in EVALUATION_SCOPES:
        raise ValueError(
            "evaluation_scope must be one of: " + ", ".join(EVALUATION_SCOPES)
        )
    if restore_checkpoint is not None and not restore_checkpoint.exists():
        raise ValueError(f"restore_checkpoint does not exist: {restore_checkpoint}")
    if reward_config["profile"] not in REWARD_PROFILES:
        raise ValueError(
            "reward.profile must be one of: " + ", ".join(REWARD_PROFILES)
        )
    validate_model_config(model_config)
    validate_policy_setup(policy_setup, players=players)
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


def build_training_config(
    *,
    iterations: int,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    reward_config: dict[str, Any],
    checkpoint_dir: Path,
    run_name: str,
    train_batch_size: int,
    minibatch_size: int,
    rollout_fragment_length: int,
    num_epochs: int,
    lr: float,
    num_env_runners: int,
    model_config: dict[str, Any],
    checkpoint_interval: int | None,
    evaluation_interval: int | None,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    evaluation_scope: str,
    restore_checkpoint: Path | None,
    wandb_mode: str,
    wandb_project: str,
    wandb_entity: str | None,
    show_progress: bool,
    suppress_ray_warnings: bool,
    dashboard_logging: dict[str, Any],
    continuation: dict[str, Any],
    policy_setup: dict[str, Any],
) -> dict[str, Any]:
    return {
        "game": {
            "players": players,
            "seed": seed,
            "max_turns": max_turns,
            "include_cards": list(include_cards) if include_cards is not None else None,
            "exclude_cards": list(exclude_cards),
            "enabled_combo_rules": list(enabled_combo_rules),
            "reveal_opponent_card_counts": reveal_opponent_card_counts,
        },
        "reward": copy.deepcopy(reward_config),
        "training": {
            "iterations": iterations,
            "train_batch_size": train_batch_size,
            "minibatch_size": minibatch_size,
            "rollout_fragment_length": rollout_fragment_length,
            "num_epochs": num_epochs,
            "lr": lr,
            "num_env_runners": num_env_runners,
        },
        "model": copy.deepcopy(model_config),
        "experiment": {
            "checkpoint_dir": str(checkpoint_dir),
            "run_name": run_name,
            "checkpoint_interval": checkpoint_interval,
            "evaluation_interval": evaluation_interval,
            "evaluation_episodes": evaluation_episodes,
            "evaluation_opponents": list(evaluation_opponents),
            "evaluation_scope": evaluation_scope,
            "restore_checkpoint": (
                str(restore_checkpoint) if restore_checkpoint is not None else None
            ),
            "wandb_mode": wandb_mode,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
            "show_progress": show_progress,
            "suppress_ray_warnings": suppress_ray_warnings,
        },
        "dashboard_logging": copy.deepcopy(dashboard_logging),
        "continuation": copy.deepcopy(continuation),
        "policy_setup": policy_setup,
    }


def training_config_to_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    payload["algorithm"] = "RLlib PPO"
    payload["policy_mapping"] = "shared_policy"
    return payload


def load_training_config(path: Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    if path is None:
        return config
    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"RLlib training config must be a mapping: {path}")
    return deep_merge_config(config, loaded)


def deep_merge_config(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key not in merged:
            raise ValueError(f"Unknown RLlib training config section or key: {key}")
        if isinstance(merged[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {key!r} must be a mapping.")
            for nested_key, nested_value in value.items():
                if nested_key not in merged[key]:
                    raise ValueError(
                        f"Unknown RLlib training config key: {key}.{nested_key}"
                    )
                merged[key][nested_key] = nested_value
        else:
            merged[key] = value
    return merged


def apply_cli_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    overrides: tuple[tuple[str, str, Any], ...] = (
        ("training", "iterations", args.iterations),
        ("game", "players", args.players),
        ("game", "seed", args.seed),
        ("game", "max_turns", args.max_turns),
        ("game", "include_cards", args.include_cards),
        ("game", "exclude_cards", args.exclude_cards),
        ("game", "enabled_combo_rules", args.enabled_combo_rules),
        (
            "game",
            "reveal_opponent_card_counts",
            reveal_opponent_card_counts_override(args),
        ),
        ("experiment", "checkpoint_dir", args.checkpoint_dir),
        ("experiment", "run_name", args.run_name),
        ("training", "train_batch_size", args.train_batch_size),
        ("training", "minibatch_size", args.minibatch_size),
        ("training", "rollout_fragment_length", args.rollout_fragment_length),
        ("training", "num_epochs", args.num_epochs),
        ("training", "lr", args.lr),
        ("training", "num_env_runners", args.num_env_runners),
        ("model", "hidden_layers", args.model_hidden_layers),
        ("model", "activation", args.model_activation),
        ("experiment", "checkpoint_interval", args.checkpoint_interval),
        ("experiment", "evaluation_interval", args.evaluation_interval),
        ("experiment", "evaluation_episodes", args.evaluation_episodes),
        ("experiment", "evaluation_opponents", args.evaluation_opponents),
        ("experiment", "evaluation_scope", args.evaluation_scope),
        ("experiment", "restore_checkpoint", args.restore_checkpoint),
        ("experiment", "existing_run_action", args.existing_run_action),
        ("experiment", "continue_mode", args.continue_mode),
        ("experiment", "wandb_mode", args.wandb_mode),
        ("experiment", "wandb_project", args.wandb_project),
        ("experiment", "wandb_entity", args.wandb_entity),
        ("experiment", "show_progress", show_progress_override(args)),
        ("experiment", "suppress_ray_warnings", suppress_ray_warnings_override(args)),
        ("dashboard_logging", "enabled", dashboard_logging_enabled_override(args)),
        ("dashboard_logging", "experiment_dir", args.experiment_dir),
        ("dashboard_logging", "run_name", args.dashboard_run_name),
    )
    for section, key, value in overrides:
        if value is not None:
            merged[section][key] = value
    return merged


def normalize_training_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    game = normalized["game"]
    reward = normalized["reward"]
    training = normalized["training"]
    model = normalized["model"]
    experiment = normalized["experiment"]
    game["players"] = int(game["players"])
    game["seed"] = int(game["seed"])
    game["max_turns"] = int(game["max_turns"])
    game["include_cards"] = normalize_optional_name_sequence(game["include_cards"])
    game["exclude_cards"] = normalize_name_sequence(game["exclude_cards"])
    game["enabled_combo_rules"] = normalize_name_sequence(game["enabled_combo_rules"])
    game["reveal_opponent_card_counts"] = bool(game["reveal_opponent_card_counts"])
    training["iterations"] = int(training["iterations"])
    training["train_batch_size"] = int(training["train_batch_size"])
    training["minibatch_size"] = int(training["minibatch_size"])
    training["rollout_fragment_length"] = int(training["rollout_fragment_length"])
    training["num_epochs"] = int(training["num_epochs"])
    training["lr"] = float(training["lr"])
    training["num_env_runners"] = int(training["num_env_runners"])
    normalized["model"] = normalize_model_config(model)
    experiment["checkpoint_dir"] = str(experiment["checkpoint_dir"])
    experiment["run_name"] = str(experiment["run_name"])
    experiment["checkpoint_interval"] = normalize_optional_positive_int(
        experiment["checkpoint_interval"]
    )
    experiment["evaluation_interval"] = normalize_optional_positive_int(
        experiment["evaluation_interval"]
    )
    experiment["evaluation_episodes"] = int(experiment["evaluation_episodes"])
    experiment["evaluation_opponents"] = normalize_name_sequence(
        experiment["evaluation_opponents"]
    )
    experiment["evaluation_scope"] = str(experiment["evaluation_scope"])
    experiment["restore_checkpoint"] = (
        str(experiment["restore_checkpoint"])
        if experiment["restore_checkpoint"] is not None
        else None
    )
    experiment["existing_run_action"] = str(experiment["existing_run_action"])
    if experiment["existing_run_action"] not in EXISTING_RUN_ACTIONS:
        raise ValueError(
            "existing_run_action must be one of: " + ", ".join(EXISTING_RUN_ACTIONS)
        )
    experiment["continue_mode"] = str(experiment["continue_mode"])
    if experiment["continue_mode"] not in CONTINUE_MODES:
        raise ValueError(
            "continue_mode must be one of: " + ", ".join(CONTINUE_MODES)
        )
    experiment["wandb_mode"] = str(experiment["wandb_mode"])
    experiment["wandb_project"] = str(experiment["wandb_project"])
    if experiment["wandb_entity"] is not None:
        experiment["wandb_entity"] = str(experiment["wandb_entity"])
    experiment["show_progress"] = bool(experiment["show_progress"])
    experiment["suppress_ray_warnings"] = bool(experiment["suppress_ray_warnings"])
    normalized["reward"] = normalize_reward_config(
        reward,
        players=game["players"],
    )
    normalized["dashboard_logging"] = normalize_dashboard_logging(
        normalized.get("dashboard_logging", {}),
        fallback_run_name=experiment["run_name"],
    )
    normalized["policy_setup"] = normalize_policy_setup(
        normalized.get("policy_setup", {}),
        players=game["players"],
    )
    normalized["continuation"] = normalize_continuation_config(
        normalized.get("continuation", {})
    )
    return normalized


def normalize_model_config(model_config: dict[str, Any]) -> dict[str, Any]:
    default = DEFAULT_TRAINING_CONFIG["model"]
    config = copy.deepcopy(default)
    config.update(model_config)
    hidden_layers = normalize_positive_int_sequence(config["hidden_layers"])
    activation = str(config["activation"])
    normalized = {
        "hidden_layers": hidden_layers,
        "activation": activation,
    }
    validate_model_config(normalized)
    return normalized


def validate_model_config(model_config: dict[str, Any]) -> None:
    hidden_layers = model_config.get("hidden_layers")
    if not isinstance(hidden_layers, list):
        raise ValueError("model.hidden_layers must be a list of positive integers.")
    if any(not isinstance(size, int) or size < 1 for size in hidden_layers):
        raise ValueError("model.hidden_layers must contain positive integers.")
    activation = str(model_config.get("activation"))
    if activation not in MODEL_ACTIVATIONS:
        raise ValueError(
            "model.activation must be one of: " + ", ".join(MODEL_ACTIVATIONS)
        )


def normalize_positive_int_sequence(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        return list(parse_int_list(value))
    return [int(item) for item in value]


def normalize_reward_config(
    reward_config: dict[str, Any],
    *,
    players: int,
) -> dict[str, Any]:
    default = DEFAULT_TRAINING_CONFIG["reward"]
    config = copy.deepcopy(default)
    config.update(reward_config)
    profile = str(config.get("profile") or default["profile"])
    if profile not in REWARD_PROFILES:
        raise ValueError("reward.profile must be one of: " + ", ".join(REWARD_PROFILES))
    terminal_rank_rewards = normalize_terminal_rank_rewards(
        players,
        config.get("terminal_rank_rewards"),
    )
    return {
        "profile": profile,
        "terminal_rank_rewards": list(terminal_rank_rewards),
    }


def normalize_terminal_rank_rewards(
    players: int,
    rewards: Any,
) -> tuple[float, ...]:
    if rewards is None:
        return default_terminal_rank_rewards(players)
    normalized = tuple(float(reward) for reward in rewards)
    if len(normalized) != players:
        raise ValueError(
            "reward.terminal_rank_rewards must contain exactly "
            f"{players} value(s), one for each finishing rank."
        )
    return normalized


def default_terminal_rank_rewards(players: int) -> tuple[float, ...]:
    if players == 2:
        return (1.0, -1.0)
    if players == 4:
        return (1.0, 0.3, -0.3, -1.0)
    step = 2.0 / (players - 1)
    return tuple(1.0 - rank_index * step for rank_index in range(players))


def normalize_policy_setup(
    policy_setup: dict[str, Any],
    *,
    players: int,
) -> dict[str, Any]:
    setup = copy.deepcopy(DEFAULT_TRAINING_CONFIG["policy_setup"])
    setup = deep_merge_config({"policy_setup": setup}, {"policy_setup": policy_setup})[
        "policy_setup"
    ]
    mode = str(setup["mode"])
    setup["mode"] = mode
    setup["shared_policy_id"] = str(setup.get("shared_policy_id") or SHARED_POLICY_ID)
    setup["opponent_pool"] = normalize_opponent_pool_config(setup["opponent_pool"])

    if mode == "shared":
        setup["seat_policies"] = {
            f"player_{index}": setup["shared_policy_id"]
            for index in range(1, players + 1)
        }
        setup["trainable_policies"] = [setup["shared_policy_id"]]
        setup["frozen_policies"] = {}
        return setup

    if mode == "separate-per-seat":
        setup["seat_policies"] = {
            f"player_{index}": f"player_{index}_policy"
            for index in range(1, players + 1)
        }
        setup["trainable_policies"] = list(setup["seat_policies"].values())
        setup["frozen_policies"] = {}
        return setup

    if mode == "learner-vs-frozen":
        setup["seat_policies"] = normalize_string_mapping(setup["seat_policies"])
        setup["trainable_policies"] = normalize_name_sequence(
            setup["trainable_policies"]
        )
        setup["frozen_policies"] = normalize_frozen_policies(setup["frozen_policies"])
        return setup

    if mode == "learner-vs-pool":
        setup["seat_policies"], setup["frozen_policies"] = build_pool_policy_setup(
            setup,
            players=players,
        )
        setup["trainable_policies"] = normalize_name_sequence(
            setup["trainable_policies"]
        ) or ["learner_policy"]
        return setup

    raise ValueError(
        "policy_setup.mode must be one of: shared, separate-per-seat, "
        "learner-vs-frozen, learner-vs-pool."
    )


def normalize_opponent_pool_config(value: dict[str, Any]) -> dict[str, Any]:
    default = DEFAULT_TRAINING_CONFIG["policy_setup"]["opponent_pool"]
    return {
        "metadata_path": value.get("metadata_path", default["metadata_path"]),
        "sample_mode": str(value.get("sample_mode") or default["sample_mode"]),
        "seed": int(value.get("seed") or default["seed"]),
    }


def normalize_string_mapping(value: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(item) for key, item in value.items()}


def normalize_frozen_policies(value: dict[str, Any]) -> dict[str, dict[str, str]]:
    frozen: dict[str, dict[str, str]] = {}
    for policy_id, spec in value.items():
        if isinstance(spec, str):
            frozen[str(policy_id)] = {
                "checkpoint_path": spec,
                "policy_id": SHARED_POLICY_ID,
            }
        elif isinstance(spec, dict):
            frozen[str(policy_id)] = {
                "checkpoint_path": str(spec["checkpoint_path"]),
                "policy_id": str(spec.get("policy_id") or SHARED_POLICY_ID),
            }
        else:
            raise ValueError(f"Invalid frozen policy spec for {policy_id}.")
    return frozen


def build_pool_policy_setup(
    setup: dict[str, Any],
    *,
    players: int,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    from exploding_kittens.policy_pool import PolicyPool

    pool_config = setup["opponent_pool"]
    metadata_path = pool_config["metadata_path"]
    if metadata_path is None:
        raise ValueError("learner-vs-pool requires opponent_pool.metadata_path.")
    payload = yaml.safe_load(Path(metadata_path).read_text(encoding="utf-8")) or {}
    entries = tuple(
        pool_entry_from_metadata(entry)
        for entry in payload.get("entries", [])
    )
    pool = PolicyPool(entries)
    sampled = pool.sample_opponents(
        players=players,
        mode=pool_config["sample_mode"],
        rng=__import__("random").Random(pool_config["seed"]),
    )
    seat_policies = {"player_1": "learner_policy"}
    frozen_policies: dict[str, dict[str, str]] = {}
    for seat, policy_spec in sampled.items():
        if not policy_spec.startswith("rllib:"):
            raise ValueError(
                "learner-vs-pool training currently requires RLlib checkpoint "
                f"opponents, but sampled {policy_spec!r} for {seat}."
            )
        policy_id = f"{seat}_frozen_policy"
        seat_policies[seat] = policy_id
        frozen_policies[policy_id] = {
            "checkpoint_path": policy_spec.removeprefix("rllib:"),
            "policy_id": SHARED_POLICY_ID,
        }
    return seat_policies, frozen_policies


def pool_entry_from_metadata(entry: dict[str, Any]) -> Any:
    from exploding_kittens.policy_pool import PolicyPoolEntry

    checkpoint_path = entry.get("checkpoint_path")
    return PolicyPoolEntry(
        name=str(entry["name"]),
        policy_spec=str(entry["policy_spec"]),
        kind=str(entry["kind"]),
        role=str(entry["role"]),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        iteration=entry.get("iteration"),
        score=entry.get("score"),
    )


def validate_policy_setup(policy_setup: dict[str, Any], *, players: int) -> None:
    player_names = {f"player_{index}" for index in range(1, players + 1)}
    seat_policies = policy_setup["seat_policies"]
    missing_seats = sorted(player_names - set(seat_policies))
    unknown_seats = sorted(set(seat_policies) - player_names)
    if missing_seats:
        raise ValueError(f"policy_setup.seat_policies missing: {', '.join(missing_seats)}")
    if unknown_seats:
        raise ValueError(f"Unknown policy_setup seat(s): {', '.join(unknown_seats)}")
    policies = set(seat_policies.values())
    trainable = set(policy_setup["trainable_policies"])
    if not trainable:
        raise ValueError("policy_setup.trainable_policies cannot be empty.")
    unknown_trainable = sorted(trainable - policies)
    if unknown_trainable:
        raise ValueError(
            "policy_setup.trainable_policies contains unmapped policies: "
            + ", ".join(unknown_trainable)
        )
    frozen = set(policy_setup["frozen_policies"])
    unknown_frozen = sorted(frozen - policies)
    if unknown_frozen:
        raise ValueError(
            "policy_setup.frozen_policies contains unmapped policies: "
            + ", ".join(unknown_frozen)
        )
    overlap = sorted(trainable & frozen)
    if overlap:
        raise ValueError(
            "Policies cannot be both trainable and frozen: " + ", ".join(overlap)
        )


def normalize_name_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return list(parse_name_list(value))
    return [str(item) for item in value]


def normalize_optional_name_sequence(value: Any) -> list[str] | None:
    if value is None:
        return None
    return normalize_name_sequence(value)


def normalize_optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def normalize_dashboard_logging(
    value: dict[str, Any],
    *,
    fallback_run_name: str,
) -> dict[str, Any]:
    default = DEFAULT_TRAINING_CONFIG["dashboard_logging"]
    run_name = value.get("run_name", default["run_name"]) or fallback_run_name
    return {
        "enabled": bool(value.get("enabled", default["enabled"])),
        "experiment_dir": str(
            value.get("experiment_dir") or default["experiment_dir"]
        ),
        "run_name": str(run_name),
        "flush_each_iteration": bool(
            value.get("flush_each_iteration", default["flush_each_iteration"])
        ),
    }


def normalize_continuation_config(value: dict[str, Any]) -> dict[str, Any]:
    default = DEFAULT_TRAINING_CONFIG["continuation"]
    config = copy.deepcopy(default)
    config.update(value)
    return {
        "parent_run_name": (
            str(config["parent_run_name"])
            if config.get("parent_run_name") is not None
            else None
        ),
        "restore_checkpoint": (
            str(config["restore_checkpoint"])
            if config.get("restore_checkpoint") is not None
            else None
        ),
        "start_iteration": int(config.get("start_iteration") or 0),
        "mode": str(config["mode"]) if config.get("mode") is not None else None,
        "wandb_run_id": (
            str(config["wandb_run_id"])
            if config.get("wandb_run_id") is not None
            else None
        ),
    }


def save_resolved_config(path: Path, config: dict[str, Any]) -> None:
    config_to_save = copy.deepcopy(config)
    config_to_save.pop("algorithm", None)
    config_to_save.pop("policy_mapping", None)
    with path.open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config_to_save, config_file, sort_keys=False)


def prepare_config_for_existing_run(
    config: dict[str, Any],
    *,
    input_fn: Any = input,
    output_fn: Any = print,
    interactive: bool | None = None,
) -> dict[str, Any] | None:
    normalized = normalize_training_config(config)
    info = inspect_existing_run(normalized)
    if not info.exists:
        return normalized

    action = normalized["experiment"]["existing_run_action"]
    if action not in EXISTING_RUN_ACTIONS:
        raise ValueError(
            "existing_run_action must be one of: " + ", ".join(EXISTING_RUN_ACTIONS)
        )
    if action == "ask":
        if interactive is None:
            interactive = sys.stdin.isatty()
        if not interactive:
            output_existing_run_notice(info, output_fn=output_fn)
            output_fn("Existing run action is ask, but stdin is not interactive. Quitting.")
            return None
        action = prompt_existing_run_action(info, input_fn=input_fn, output_fn=output_fn)

    if action == "quit":
        output_fn("No training started.")
        return None
    if action == "continue":
        if info.latest_checkpoint is None:
            output_fn("Cannot continue: no checkpoint was found.")
            return None
        if info.config_compatibility == "incompatible":
            output_fn(
                "Cannot continue: previous config is incompatible. "
                "Changed field(s): " + ", ".join(info.incompatible_fields)
            )
            return None
        return build_continuation_config(normalized, info)
    if action == "overwrite":
        if not confirm_overwrite(info, input_fn=input_fn, output_fn=output_fn):
            output_fn("Overwrite cancelled. No training started.")
            return None
        remove_existing_run_outputs(info)
        return normalized
    raise ValueError(
        "existing_run_action must be one of: " + ", ".join(EXISTING_RUN_ACTIONS)
    )


def inspect_existing_run(config: dict[str, Any]) -> ExistingRunInfo:
    experiment = config["experiment"]
    dashboard_logging = config["dashboard_logging"]
    run_name = experiment["run_name"]
    checkpoint_dir = Path(experiment["checkpoint_dir"]) / run_name
    dashboard_dir = Path(dashboard_logging["experiment_dir"]) / dashboard_logging["run_name"]
    exists = checkpoint_dir.exists() or dashboard_dir.exists()
    previous_config = load_previous_resolved_config(checkpoint_dir, dashboard_dir)
    compatibility, incompatible_fields = config_compatibility(previous_config, config)
    return ExistingRunInfo(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        dashboard_dir=dashboard_dir,
        exists=exists,
        latest_checkpoint=latest_checkpoint_path(checkpoint_dir),
        previous_status=previous_run_status(dashboard_dir),
        config_compatibility=compatibility,
        incompatible_fields=incompatible_fields,
        wandb_run_id=(
            previous_wandb_run_id(dashboard_dir)
            or previous_wandb_run_id_from_local_wandb(run_name)
        ),
    )


def output_existing_run_notice(
    info: ExistingRunInfo,
    *,
    output_fn: Any = print,
) -> None:
    latest = info.latest_checkpoint.name if info.latest_checkpoint else "none"
    output_fn(f"Existing training run found: {info.run_name}")
    output_fn("")
    output_fn(f"Dashboard logs: {info.dashboard_dir}")
    output_fn(f"Checkpoints:     {info.checkpoint_dir}")
    output_fn(f"Latest checkpoint: {latest}")
    output_fn(f"Previous status: {info.previous_status}")
    output_fn(f"Previous config: {info.config_compatibility}")
    if info.incompatible_fields:
        output_fn("Changed field(s): " + ", ".join(info.incompatible_fields))
    output_fn("")


def prompt_existing_run_action(
    info: ExistingRunInfo,
    *,
    input_fn: Any = input,
    output_fn: Any = print,
) -> str:
    output_existing_run_notice(info, output_fn=output_fn)
    choices = ["[o] Overwrite this run", "[q] Quit"]
    if info.latest_checkpoint is not None and info.config_compatibility != "incompatible":
        choices.insert(0, f"[c] Continue training from {info.latest_checkpoint.name}")
    output_fn("What would you like to do?")
    output_fn("")
    for choice in choices:
        output_fn(choice)
    output_fn("")
    while True:
        choice = input_fn("Choice: ").strip().lower()
        if choice in {"q", "quit"}:
            return "quit"
        if choice in {"o", "overwrite"}:
            return "overwrite"
        if (
            choice in {"c", "continue"}
            and info.latest_checkpoint is not None
            and info.config_compatibility != "incompatible"
        ):
            return "continue"
        output_fn("Please choose one of: c, o, q.")


def confirm_overwrite(
    info: ExistingRunInfo,
    *,
    input_fn: Any = input,
    output_fn: Any = print,
) -> bool:
    phrase = f"OVERWRITE {info.run_name}"
    output_fn("Overwrite will delete existing dashboard logs and checkpoints.")
    output_fn(f"Type {phrase!r} to confirm:")
    return input_fn("Confirm: ").strip() == phrase


def build_continuation_config(
    config: dict[str, Any],
    info: ExistingRunInfo,
) -> dict[str, Any]:
    if info.latest_checkpoint is None:
        raise ValueError("Cannot continue without a checkpoint.")
    continued = copy.deepcopy(config)
    continue_mode = continued["experiment"]["continue_mode"]
    new_run_name = (
        info.run_name
        if continue_mode == "same-run"
        else next_continuation_run_name(config, info.run_name)
    )
    continued["experiment"]["run_name"] = new_run_name
    continued["experiment"]["restore_checkpoint"] = str(info.latest_checkpoint)
    continued["experiment"]["existing_run_action"] = "ask"
    continued["dashboard_logging"]["run_name"] = new_run_name
    start_iteration = checkpoint_iteration(info.latest_checkpoint) or 0
    continued["continuation"] = {
        "parent_run_name": info.run_name,
        "restore_checkpoint": str(info.latest_checkpoint),
        "start_iteration": start_iteration,
        "mode": continue_mode,
        "wandb_run_id": info.wandb_run_id,
    }
    return continued


def next_continuation_run_name(config: dict[str, Any], run_name: str) -> str:
    checkpoint_root = Path(config["experiment"]["checkpoint_dir"])
    experiment_root = Path(config["dashboard_logging"]["experiment_dir"])
    for index in range(1, 1000):
        candidate = f"{run_name}_continue_{index:03d}"
        if not (checkpoint_root / candidate).exists() and not (
            experiment_root / candidate
        ).exists():
            return candidate
    raise ValueError(f"Could not find an available continuation name for {run_name}.")


def remove_existing_run_outputs(info: ExistingRunInfo) -> None:
    for path in {info.checkpoint_dir, info.dashboard_dir}:
        if not path.exists():
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    numbered = [
        path
        for path in checkpoint_dir.iterdir()
        if path.is_dir() and checkpoint_iteration(path) is not None
    ]
    if numbered:
        return max(numbered, key=lambda path: checkpoint_iteration(path) or -1)
    return checkpoint_dir if checkpoint_dir.is_dir() else None


def checkpoint_iteration(path: Path) -> int | None:
    name = path.name
    if not name.startswith("checkpoint_"):
        return None
    try:
        return int(name.removeprefix("checkpoint_"))
    except ValueError:
        return None


def previous_run_status(dashboard_dir: Path) -> str:
    summary_path = dashboard_dir / "summary.json"
    if not summary_path.exists():
        return "unknown"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    return str(summary.get("status") or "unknown")


def previous_wandb_run_id(dashboard_dir: Path) -> str | None:
    summary_path = dashboard_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    wandb_run_id = summary.get("wandb_run_id")
    return str(wandb_run_id) if wandb_run_id else None


def previous_wandb_run_id_from_local_wandb(
    run_name: str,
    *,
    wandb_dir: Path = Path("wandb"),
) -> str | None:
    if not wandb_dir.exists():
        return None
    candidates: list[tuple[float, str]] = []
    for run_dir in wandb_dir.iterdir():
        if not run_dir.is_dir():
            continue
        wandb_run_id = wandb_run_id_from_directory_name(run_dir.name)
        if wandb_run_id is None:
            continue
        config_path = run_dir / "files" / "config.yaml"
        if not config_path.exists():
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        if wandb_config_run_name(config) == run_name:
            candidates.append((run_dir.stat().st_mtime, wandb_run_id))
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate[0])[1]


def wandb_run_id_from_directory_name(name: str) -> str | None:
    if name.startswith("run-") or name.startswith("offline-run-"):
        return name.rsplit("-", maxsplit=1)[-1] or None
    return None


def wandb_config_run_name(config: dict[str, Any]) -> str | None:
    experiment = wandb_config_value(config, "experiment")
    if isinstance(experiment, dict) and experiment.get("run_name") is not None:
        return str(experiment["run_name"])
    dashboard_logging = wandb_config_value(config, "dashboard_logging")
    if (
        isinstance(dashboard_logging, dict)
        and dashboard_logging.get("run_name") is not None
    ):
        return str(dashboard_logging["run_name"])
    return None


def wandb_config_value(config: dict[str, Any], key: str) -> Any:
    value = config.get(key)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def load_previous_resolved_config(
    checkpoint_dir: Path,
    dashboard_dir: Path,
) -> dict[str, Any] | None:
    for path in (
        dashboard_dir / RESOLVED_CONFIG_FILENAME,
        checkpoint_dir / RESOLVED_CONFIG_FILENAME,
    ):
        if path.exists():
            with path.open("r", encoding="utf-8") as config_file:
                loaded = yaml.safe_load(config_file) or {}
            return loaded if isinstance(loaded, dict) else None
    return None


def config_compatibility(
    previous_config: dict[str, Any] | None,
    current_config: dict[str, Any],
) -> tuple[str, tuple[str, ...]]:
    if previous_config is None:
        return "unknown", ()
    try:
        previous = normalize_training_config(
            deep_merge_config(DEFAULT_TRAINING_CONFIG, previous_config)
        )
    except ValueError:
        return "unknown", ()
    important_fields = ("game", "reward", "model", "policy_setup")
    changed = tuple(
        field for field in important_fields if previous[field] != current_config[field]
    )
    if changed:
        return "incompatible", changed
    return "compatible", ()


def train_from_config(config: dict[str, Any]) -> RLLibTrainingResult:
    normalized = normalize_training_config(config)
    game = normalized["game"]
    reward = normalized["reward"]
    training = normalized["training"]
    model = normalized["model"]
    experiment = normalized["experiment"]
    continuation = normalized["continuation"]
    return train_rllib_shared_policy(
        iterations=training["iterations"],
        players=game["players"],
        seed=game["seed"],
        max_turns=game["max_turns"],
        include_cards=tuple(game["include_cards"]) if game["include_cards"] else None,
        exclude_cards=tuple(game["exclude_cards"]),
        enabled_combo_rules=tuple(game["enabled_combo_rules"]),
        reveal_opponent_card_counts=game["reveal_opponent_card_counts"],
        reward_config=reward,
        checkpoint_dir=Path(experiment["checkpoint_dir"]),
        run_name=experiment["run_name"],
        train_batch_size=training["train_batch_size"],
        minibatch_size=training["minibatch_size"],
        rollout_fragment_length=training["rollout_fragment_length"],
        num_epochs=training["num_epochs"],
        lr=training["lr"],
        num_env_runners=training["num_env_runners"],
        model_config=model,
        checkpoint_interval=experiment["checkpoint_interval"],
        evaluation_interval=experiment["evaluation_interval"],
        evaluation_episodes=experiment["evaluation_episodes"],
        evaluation_opponents=tuple(experiment["evaluation_opponents"]),
        evaluation_scope=experiment["evaluation_scope"],
        restore_checkpoint=(
            Path(experiment["restore_checkpoint"])
            if experiment["restore_checkpoint"] is not None
            else None
        ),
        wandb_mode=experiment["wandb_mode"],
        wandb_project=experiment["wandb_project"],
        wandb_entity=experiment["wandb_entity"],
        show_progress=experiment["show_progress"],
        suppress_ray_warnings=experiment["suppress_ray_warnings"],
        dashboard_logging=normalized["dashboard_logging"],
        continuation=continuation,
        policy_setup=normalized["policy_setup"],
    )


def should_save_iteration_checkpoint(
    iteration: int,
    checkpoint_interval: int | None,
    evaluation_interval: int | None,
) -> bool:
    if iteration == 1 and (checkpoint_interval is not None or evaluation_interval is not None):
        return True
    return checkpoint_interval is not None and iteration % checkpoint_interval == 0


def should_evaluate_iteration(
    iteration: int,
    evaluation_interval: int | None,
) -> bool:
    return evaluation_interval is not None and iteration % evaluation_interval == 0


def iteration_checkpoint_name(iteration: int) -> str:
    return f"checkpoint_{iteration:06d}"


def save_algorithm_checkpoint(
    algorithm: Any,
    checkpoint_path: Path,
    config_payload: dict[str, Any],
) -> Path:
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    save_resolved_config(
        checkpoint_path / RESOLVED_CONFIG_FILENAME,
        config_payload,
    )
    algorithm.save(str(checkpoint_path.resolve()))
    return checkpoint_path.resolve()


def evaluate_training_checkpoint(
    *,
    checkpoint_path: Path,
    iteration: int,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    evaluation_rank_rewards: tuple[float, ...],
    policy_setup: dict[str, Any],
    evaluation_scope: str,
) -> dict[str, Any]:
    if evaluation_scope == RANDOMIZED_SEAT_RANK_EVALUATION:
        return evaluate_checkpoint_by_randomized_seat_rank(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            players=players,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            evaluation_episodes=evaluation_episodes,
            evaluation_opponents=evaluation_opponents,
            evaluation_rank_rewards=evaluation_rank_rewards,
            policy_setup=policy_setup,
        )
    if evaluation_scope == NATIVE_SEAT_EVALUATION:
        return evaluate_checkpoint_by_native_seat(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            players=players,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            evaluation_episodes=evaluation_episodes,
            evaluation_opponents=evaluation_opponents,
            policy_setup=policy_setup,
        )
    if evaluation_scope == PRIMARY_SEAT_EVALUATION:
        return evaluate_checkpoint_against_scripted_baselines(
            checkpoint_path=checkpoint_path,
            iteration=iteration,
            players=players,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            evaluation_episodes=evaluation_episodes,
            evaluation_opponents=evaluation_opponents,
            policy_setup=policy_setup,
        )
    raise ValueError(
        "evaluation_scope must be one of: " + ", ".join(EVALUATION_SCOPES)
    )


def evaluate_checkpoint_against_scripted_baselines(
    *,
    checkpoint_path: Path,
    iteration: int,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    policy_setup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from scripts.evaluate_rllib_checkpoint import evaluate_rllib_checkpoint

    normalized_policy_setup = normalize_policy_setup(policy_setup or {}, players=players)
    rllib_policy_id = normalized_policy_setup["seat_policies"]["player_1"]
    results: dict[str, Any] = {}
    for opponent in evaluation_opponents:
        seat_policies = {
            "player_1": f"rllib:{checkpoint_path}",
            **{
                f"player_{player_index}": opponent
                for player_index in range(2, players + 1)
            },
        }
        results[opponent] = evaluate_rllib_checkpoint(
            seat_policies=seat_policies,
            episodes=evaluation_episodes,
            players=players,
            seed=seed + iteration * 1000,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            wandb_mode="disabled",
            policy_id=rllib_policy_id,
        )
    return results


def evaluate_checkpoint_by_native_seat(
    *,
    checkpoint_path: Path,
    iteration: int,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    policy_setup: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    from scripts.evaluate_rllib_checkpoint import evaluate_rllib_checkpoint

    normalized_policy_setup = normalize_policy_setup(policy_setup, players=players)
    trainable_policy_ids = set(normalized_policy_setup["trainable_policies"])
    native_seats = {
        seat: policy_id
        for seat, policy_id in normalized_policy_setup["seat_policies"].items()
        if policy_id in trainable_policy_ids
    }
    results: dict[str, dict[str, Any]] = {}
    for opponent in evaluation_opponents:
        opponent_results: dict[str, Any] = {}
        for seat, policy_id in native_seats.items():
            seat_policies = {
                f"player_{player_index}": opponent
                for player_index in range(1, players + 1)
            }
            seat_policies[seat] = f"rllib:{checkpoint_path}"
            opponent_results[seat] = evaluate_rllib_checkpoint(
                seat_policies=seat_policies,
                episodes=evaluation_episodes,
                players=players,
                seed=native_seat_evaluation_seed(seed, iteration, opponent, seat),
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                wandb_mode="disabled",
                policy_id=policy_id,
            )
        results[opponent] = opponent_results
    return results


def evaluate_checkpoint_by_randomized_seat_rank(
    *,
    checkpoint_path: Path,
    iteration: int,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    evaluation_episodes: int,
    evaluation_opponents: tuple[str, ...],
    evaluation_rank_rewards: tuple[float, ...],
    policy_setup: dict[str, Any],
) -> dict[str, RandomizedSeatRankEvaluationResult]:
    normalized_policy_setup = normalize_policy_setup(policy_setup, players=players)
    trainable_policy_ids = tuple(normalized_policy_setup["trainable_policies"])
    results: dict[str, RandomizedSeatRankEvaluationResult] = {}
    for policy_id in trainable_policy_ids:
        results[policy_id] = evaluate_policy_by_randomized_seat_rank(
            checkpoint_path=checkpoint_path,
            policy_id=policy_id,
            players=players,
            seed=randomized_seat_rank_evaluation_seed(seed, iteration, policy_id),
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            episodes=evaluation_episodes,
            opponent_pool=evaluation_opponents,
            rank_rewards=evaluation_rank_rewards,
        )
    return results


def evaluate_policy_by_randomized_seat_rank(
    *,
    checkpoint_path: Path,
    policy_id: str,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    episodes: int,
    opponent_pool: tuple[str, ...],
    rank_rewards: tuple[float, ...],
) -> RandomizedSeatRankEvaluationResult:
    from scripts.evaluate_multi_policy_game import (
        RandomMaskController,
        ScriptedAgentController,
        run_multi_policy_episode,
    )
    from scripts.evaluate_rllib_checkpoint import RLLibController, load_rllib_policy

    validate_randomized_seat_rank_args(
        players=players,
        episodes=episodes,
        opponent_pool=opponent_pool,
        rank_rewards=rank_rewards,
    )
    player_names = tuple(f"player_{index + 1}" for index in range(players))
    seat_schedule = balanced_evaluation_seats(
        player_names=player_names,
        episodes=episodes,
        seed=seed,
    )
    policy = load_rllib_policy(checkpoint_path, policy_id=policy_id)
    rng = random.Random(seed)
    wins = 0
    rank_reward_total = 0.0
    finish_position_total = 0.0
    place_counts: Counter[int] = Counter()
    seat_counts: Counter[str] = Counter()
    seat_wins: Counter[str] = Counter()
    opponent_counts: Counter[str] = Counter()

    for episode_index, rllib_seat in enumerate(seat_schedule):
        controllers: dict[str, Any] = {}
        for seat in player_names:
            if seat == rllib_seat:
                controllers[seat] = RLLibController(
                    policy=policy,
                    policy_label=f"rllib:{checkpoint_path.name}:{policy_id}",
                )
                continue
            opponent = rng.choice(opponent_pool)
            opponent_counts[opponent] += 1
            if opponent == "random":
                controllers[seat] = RandomMaskController()
            else:
                controllers[seat] = ScriptedAgentController(opponent)

        episode = run_multi_policy_episode(
            episode=episode_index,
            player_names=player_names,
            controllers=controllers,
            seed=seed + episode_index,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
        )
        ranked_players = ranked_players_for_episode(
            player_names=player_names,
            winner=episode.winner,
            elimination_order=episode.elimination_order,
        )
        finish_position = ranked_players.index(rllib_seat) + 1
        rank_reward = rank_rewards[finish_position - 1]
        won = episode.winner == rllib_seat
        wins += int(won)
        rank_reward_total += rank_reward
        finish_position_total += finish_position
        place_counts[finish_position] += 1
        seat_counts[rllib_seat] += 1
        seat_wins[rllib_seat] += int(won)

    return RandomizedSeatRankEvaluationResult(
        policy_id=policy_id,
        episodes=episodes,
        wins=wins,
        win_rate=wins / episodes,
        average_rank_reward=rank_reward_total / episodes,
        average_finish_position=finish_position_total / episodes,
        place_counts={place: place_counts[place] for place in range(1, players + 1)},
        place_rates={
            place: place_counts[place] / episodes for place in range(1, players + 1)
        },
        seat_counts={seat: seat_counts[seat] for seat in player_names},
        seat_win_rates={
            seat: (
                seat_wins[seat] / seat_counts[seat]
                if seat_counts[seat] > 0
                else 0.0
            )
            for seat in player_names
        },
        opponent_counts=dict(sorted(opponent_counts.items())),
        rank_rewards=rank_rewards,
    )


def validate_randomized_seat_rank_args(
    *,
    players: int,
    episodes: int,
    opponent_pool: tuple[str, ...],
    rank_rewards: tuple[float, ...],
) -> None:
    if episodes < 1:
        raise ValueError("evaluation_episodes must be at least 1.")
    if not opponent_pool:
        raise ValueError("evaluation_opponents must include at least one opponent.")
    if len(rank_rewards) != players:
        raise ValueError(
            "evaluation_rank_rewards must contain exactly "
            f"{players} value(s), one for each finishing rank."
        )


def balanced_evaluation_seats(
    *,
    player_names: tuple[str, ...],
    episodes: int,
    seed: int,
) -> tuple[str, ...]:
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    schedule = [
        player_names[index % len(player_names)]
        for index in range(episodes)
    ]
    rng = random.Random(seed)
    rng.shuffle(schedule)
    return tuple(schedule)


def ranked_players_for_episode(
    *,
    player_names: tuple[str, ...],
    winner: str | None,
    elimination_order: tuple[str, ...],
) -> tuple[str, ...]:
    if winner is not None:
        ranked = [winner]
        ranked.extend(
            player for player in reversed(elimination_order) if player != winner
        )
    else:
        ranked = [
            player for player in player_names if player not in set(elimination_order)
        ]
        ranked.extend(player for player in reversed(elimination_order))
    ranked.extend(player for player in player_names if player not in ranked)
    return tuple(ranked[: len(player_names)])


def randomized_seat_rank_evaluation_seed(
    seed: int,
    iteration: int,
    policy_id: str,
) -> int:
    policy_offset = sum(ord(character) for character in policy_id)
    return seed + iteration * 1000 + policy_offset * 10


def native_seat_evaluation_seed(
    seed: int,
    iteration: int,
    opponent: str,
    seat: str,
) -> int:
    opponent_offset = sum(ord(character) for character in opponent)
    seat_number = int(seat.rsplit("_", maxsplit=1)[-1])
    return seed + iteration * 1000 + opponent_offset * 10 + seat_number


def log_training_evaluations(
    run: Any | None,
    iteration: int,
    evaluation_results: dict[str, Any],
) -> None:
    if run is None:
        return
    if is_randomized_seat_rank_evaluation_results(evaluation_results):
        payload: dict[str, float | int] = {}
        for policy_id, result in evaluation_results.items():
            prefix = f"general_eval/{policy_id}"
            payload[f"{prefix}/win_rate"] = result.win_rate
            payload[f"{prefix}/average_rank_reward"] = result.average_rank_reward
            payload[f"{prefix}/average_finish_position"] = (
                result.average_finish_position
            )
            for place, rate in result.place_rates.items():
                payload[f"{prefix}/place_{place}_rate"] = rate
            for seat, rate in result.seat_win_rates.items():
                payload[f"{prefix}/seat_win_rate/{seat}"] = rate
        if payload:
            run.log(payload, step=iteration)
        return
    if is_native_seat_evaluation_results(evaluation_results):
        payload: dict[str, float | int] = {}
        for opponent, seat_results in evaluation_results.items():
            for seat, result in seat_results.items():
                prefix = f"eval_by_player/{opponent}/{seat}"
                payload[f"{prefix}/win_rate"] = result.rllib_combined_win_rate
                payload[f"{prefix}/scripted_opponent_win_rate"] = (
                    result.scripted_opponent_win_rate
                )
                payload[f"{prefix}/illegal_action_rate"] = result.illegal_action_rate
                payload[f"{prefix}/combo_count"] = result.combo_count
        if payload:
            run.log(payload, step=iteration)
        return
    for opponent, result in evaluation_results.items():
        prefix = f"rllib_eval_during_training/{opponent}"
        payload = {
            f"{prefix}/rllib_combined_win_rate": result.rllib_combined_win_rate,
            f"{prefix}/scripted_opponent_win_rate": (
                result.scripted_opponent_win_rate
            ),
            f"{prefix}/seat_order_advantage": result.seat_order_advantage,
            f"{prefix}/illegal_action_rate": result.illegal_action_rate,
            f"{prefix}/combo_count": result.combo_count,
        }
        for action, count in result.action_distribution.items():
            payload[f"{prefix}/action_distribution/{action}"] = count
        for card, count in result.card_distribution.items():
            payload[f"{prefix}/card_distribution/{card}"] = count
        run.log(payload, step=iteration)


def log_final_vs_first_comparison(
    run: Any | None,
    *,
    final_results: dict[str, Any],
    first_results: dict[str, Any],
) -> None:
    if run is None:
        return
    payload: dict[str, float] = {}
    if is_randomized_seat_rank_evaluation_results(final_results):
        for policy_id, final_result in final_results.items():
            first_result = first_results.get(policy_id)
            if first_result is None:
                continue
            payload[
                f"rllib_eval_final_vs_first/general_eval/{policy_id}/rank_reward_delta"
            ] = (
                final_result.average_rank_reward
                - first_result.average_rank_reward
            )
            payload[
                f"rllib_eval_final_vs_first/general_eval/{policy_id}/win_rate_delta"
            ] = final_result.win_rate - first_result.win_rate
        if payload:
            run.log(payload)
        return
    if is_native_seat_evaluation_results(final_results):
        for opponent, final_seat_results in final_results.items():
            first_seat_results = first_results.get(opponent, {})
            for seat, final_result in final_seat_results.items():
                first_result = first_seat_results.get(seat)
                if first_result is None:
                    continue
                payload[
                    f"rllib_eval_final_vs_first/{opponent}/{seat}/win_rate_delta"
                ] = (
                    final_result.rllib_combined_win_rate
                    - first_result.rllib_combined_win_rate
                )
        if payload:
            run.log(payload)
        return
    for opponent, final_result in final_results.items():
        first_result = first_results.get(opponent)
        if first_result is None:
            continue
        payload[
            f"rllib_eval_final_vs_first/{opponent}/win_rate_delta"
        ] = final_result.rllib_combined_win_rate - first_result.rllib_combined_win_rate
    if payload:
        run.log(payload)


def make_local_experiment_logger(
    dashboard_logging: dict[str, Any],
    *,
    config: dict[str, Any],
    append_existing: bool = False,
) -> LocalExperimentLogger:
    if not dashboard_logging["enabled"]:
        return LocalExperimentLogger.disabled()
    return LocalExperimentLogger(
        enabled=True,
        experiment_dir=Path(dashboard_logging["experiment_dir"]),
        run_name=dashboard_logging["run_name"],
        config=config,
        flush_each_iteration=dashboard_logging["flush_each_iteration"],
        append_existing=append_existing,
    )


def build_training_metric_row(
    *,
    iteration: int,
    total_iterations: int,
    started_at: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    elapsed_seconds = max(0.0, time.perf_counter() - started_at)
    remaining_iterations = max(0, total_iterations - iteration)
    seconds_per_iteration = elapsed_seconds / max(iteration, 1)
    environment_steps = int(result.get("num_env_steps_sampled", 0))
    return {
        "iteration": iteration,
        "total_iterations": total_iterations,
        "elapsed_seconds": elapsed_seconds,
        "estimated_seconds_remaining": seconds_per_iteration * remaining_iterations,
        "environment_steps_sampled": environment_steps,
        "agent_steps_sampled": int(result.get("num_agent_steps_sampled", 0)),
        "environment_steps_per_second": (
            environment_steps / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
        "episode_reward_mean": metric_from_result(result, "episode_reward_mean"),
        "episode_length_mean": metric_from_result(result, "episode_len_mean"),
    }


def log_local_training_evaluations(
    local_logger: LocalExperimentLogger,
    *,
    iteration: int,
    evaluation_episodes: int,
    evaluation_results: dict[str, Any],
    policy_setup: dict[str, Any],
) -> None:
    if is_randomized_seat_rank_evaluation_results(evaluation_results):
        for policy_id, result in evaluation_results.items():
            local_logger.log_evaluation(
                build_randomized_seat_rank_log_row(
                    iteration=iteration,
                    evaluation_episodes=evaluation_episodes,
                    result=result,
                    policy_id=policy_id,
                )
            )
        return
    if is_native_seat_evaluation_results(evaluation_results):
        seat_to_policy = policy_setup["seat_policies"]
        for opponent, seat_results in evaluation_results.items():
            for seat, result in seat_results.items():
                local_logger.log_evaluation(
                    build_evaluation_log_row(
                        iteration=iteration,
                        evaluation_episodes=evaluation_episodes,
                        opponent=opponent,
                        result=result,
                        evaluation_scope=NATIVE_SEAT_EVALUATION,
                        evaluated_seat=seat,
                        policy_id=seat_to_policy[seat],
                        native_seat=True,
                    )
                )
        return
    primary_seat = "player_1"
    primary_policy_id = policy_setup["seat_policies"][primary_seat]
    for opponent, result in evaluation_results.items():
        local_logger.log_evaluation(
            build_evaluation_log_row(
                iteration=iteration,
                evaluation_episodes=evaluation_episodes,
                opponent=opponent,
                result=result,
                evaluation_scope=PRIMARY_SEAT_EVALUATION,
                evaluated_seat=primary_seat,
                policy_id=primary_policy_id,
                native_seat=False,
            )
        )


def final_evaluation_highlights(evaluation_results: dict[str, Any]) -> dict[str, Any]:
    if is_randomized_seat_rank_evaluation_results(evaluation_results):
        return {
            policy_id: summarize_randomized_seat_rank_result(result)
            for policy_id, result in evaluation_results.items()
        }
    if is_native_seat_evaluation_results(evaluation_results):
        return {
            opponent: {
                seat: summarize_evaluation_result(result)
                for seat, result in seat_results.items()
            }
            for opponent, seat_results in evaluation_results.items()
        }
    return {
        opponent: summarize_evaluation_result(result)
        for opponent, result in evaluation_results.items()
    }


def build_evaluation_log_row(
    *,
    iteration: int,
    evaluation_episodes: int,
    opponent: str,
    result: Any,
    evaluation_scope: str,
    evaluated_seat: str,
    policy_id: str,
    native_seat: bool,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "opponent_strategy": opponent,
        "evaluation_episodes": evaluation_episodes,
        "evaluation_scope": evaluation_scope,
        "evaluated_seat": evaluated_seat,
        "policy_id": policy_id,
        "native_seat": native_seat,
        "rllib_combined_win_rate": result.rllib_combined_win_rate,
        "scripted_opponent_win_rate": result.scripted_opponent_win_rate,
        "seat_order_advantage": result.seat_order_advantage,
        "illegal_action_rate": result.illegal_action_rate,
        "combo_count": result.combo_count,
        "action_distribution": dict(result.action_distribution),
        "card_distribution": dict(result.card_distribution),
    }


def build_randomized_seat_rank_log_row(
    *,
    iteration: int,
    evaluation_episodes: int,
    result: RandomizedSeatRankEvaluationResult,
    policy_id: str,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "opponent_strategy": "mixed-random",
        "evaluation_episodes": evaluation_episodes,
        "evaluation_scope": RANDOMIZED_SEAT_RANK_EVALUATION,
        "evaluated_seat": "balanced-random",
        "policy_id": policy_id,
        "native_seat": False,
        "rllib_combined_win_rate": result.win_rate,
        "scripted_opponent_win_rate": 1.0 - result.win_rate,
        "average_rank_reward": result.average_rank_reward,
        "average_finish_position": result.average_finish_position,
        "place_counts": dict(result.place_counts),
        "place_rates": dict(result.place_rates),
        "seat_counts": dict(result.seat_counts),
        "seat_win_rates": dict(result.seat_win_rates),
        "opponent_counts": dict(result.opponent_counts),
        "rank_rewards": list(result.rank_rewards),
    }


def summarize_evaluation_result(result: Any) -> dict[str, Any]:
    return {
        "rllib_combined_win_rate": result.rllib_combined_win_rate,
        "scripted_opponent_win_rate": result.scripted_opponent_win_rate,
        "illegal_action_rate": result.illegal_action_rate,
        "combo_count": result.combo_count,
    }


def summarize_randomized_seat_rank_result(
    result: RandomizedSeatRankEvaluationResult,
) -> dict[str, Any]:
    return {
        "win_rate": result.win_rate,
        "average_rank_reward": result.average_rank_reward,
        "average_finish_position": result.average_finish_position,
        "place_rates": dict(result.place_rates),
        "seat_win_rates": dict(result.seat_win_rates),
    }


def make_rllib_env(config: dict[str, Any]) -> Any:
    from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv

    return PettingZooEnv(
        RLLibActionMaskObservationWrapper(
            pettingzoo_env(
                players=config.get("players", 3),
                seed=config.get("seed"),
                max_turns=config.get("max_turns", 500),
                include_cards=config.get("include_cards"),
                exclude_cards=tuple(config.get("exclude_cards", ())),
                enabled_combo_rules=tuple(config.get("enabled_combo_rules", ())),
                reveal_opponent_card_counts=config.get(
                    "reveal_opponent_card_counts",
                    False,
                ),
                reward_profile=config.get("reward_profile", "sparse"),
                terminal_rank_rewards=(
                    tuple(config["terminal_rank_rewards"])
                    if config.get("terminal_rank_rewards") is not None
                    else None
                ),
            )
        )
    )


def register_rllib_env(
    *,
    env_name: str,
    players: int,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    reward_config: dict[str, Any],
) -> None:
    from ray.tune.registry import register_env

    register_env(
        env_name,
        lambda _config: make_rllib_env(
            {
                "players": players,
                "seed": seed,
                "max_turns": max_turns,
                "include_cards": include_cards,
                "exclude_cards": exclude_cards,
                "enabled_combo_rules": enabled_combo_rules,
                "reveal_opponent_card_counts": reveal_opponent_card_counts,
                "reward_profile": reward_config["profile"],
                "terminal_rank_rewards": reward_config["terminal_rank_rewards"],
            }
        ),
    )


def register_action_mask_model(model_name: str) -> None:
    from ray.rllib.examples._old_api_stack.models.action_mask_model import (
        TorchActionMaskModel,
    )
    from ray.rllib.models import ModelCatalog

    ModelCatalog.register_custom_model(model_name, TorchActionMaskModel)


def build_ppo_config(
    *,
    env_name: str,
    train_batch_size: int,
    minibatch_size: int,
    rollout_fragment_length: int,
    num_epochs: int,
    lr: float,
    num_env_runners: int = 0,
    model_name: str = DEFAULT_MODEL_NAME,
    model_config: dict[str, Any] | None = None,
    policy_setup: dict[str, Any] | None = None,
) -> Any:
    from ray.rllib.algorithms.ppo import PPOConfig

    policy_setup = (
        normalize_policy_setup({}, players=3)
        if policy_setup is None
        else copy.deepcopy(policy_setup)
    )
    model_config = normalize_model_config(model_config or {})
    policy_ids = set(policy_setup["seat_policies"].values())
    trainable_policies = set(policy_setup["trainable_policies"])
    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env_name, disable_env_checking=True)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            model={
                "custom_model": model_name,
                "fcnet_hiddens": model_config["hidden_layers"],
                "fcnet_activation": model_config["activation"],
            },
        )
        .multi_agent(
            policies=policy_ids,
            policy_mapping_fn=build_policy_mapping_fn(policy_setup["seat_policies"]),
            policies_to_train=trainable_policies,
        )
        .resources(num_gpus=0)
    )


def shared_policy_mapping(agent_id: str, episode: Any, **kwargs: Any) -> str:
    del agent_id, episode, kwargs
    return SHARED_POLICY_ID


def build_policy_mapping_fn(seat_policies: dict[str, str]) -> Any:
    mapping = dict(seat_policies)

    def policy_mapping_fn(agent_id: str, episode: Any, **kwargs: Any) -> str:
        del episode, kwargs
        try:
            return mapping[agent_id]
        except KeyError as exc:
            raise ValueError(f"No policy configured for agent {agent_id}.") from exc

    return policy_mapping_fn


def initialize_algorithm_policies(
    algorithm: Any,
    *,
    restore_checkpoint: Path | None,
    policy_setup: dict[str, Any],
) -> None:
    if restore_checkpoint is not None:
        algorithm.restore(str(restore_checkpoint.resolve()))
        return
    load_frozen_policies(algorithm, policy_setup)


def load_frozen_policies(algorithm: Any, policy_setup: dict[str, Any]) -> None:
    frozen_policies = policy_setup.get("frozen_policies", {})
    if not frozen_policies:
        return
    from ray.rllib.policy.policy import Policy

    for policy_id, frozen_spec in frozen_policies.items():
        checkpoint_path = Path(frozen_spec["checkpoint_path"])
        source_policy_id = frozen_spec.get("policy_id") or SHARED_POLICY_ID
        if not checkpoint_path.exists():
            raise ValueError(
                f"Frozen policy checkpoint does not exist for {policy_id}: "
                f"{checkpoint_path}"
            )
        restored = Policy.from_checkpoint(
            str(checkpoint_path.resolve()),
            policy_ids=[source_policy_id],
        )
        source_policy = (
            restored[source_policy_id] if isinstance(restored, dict) else restored
        )
        target_policy = algorithm.get_policy(policy_id)
        if target_policy is None:
            raise ValueError(f"RLlib did not create policy {policy_id}.")
        target_policy.set_weights(source_policy.get_weights())


def metric_from_result(result: dict[str, Any], metric_name: str) -> float | None:
    if metric_name in result and result[metric_name] == result[metric_name]:
        return float(result[metric_name])
    env_runner_metrics = result.get("env_runners", {})
    value = env_runner_metrics.get(metric_name)
    if value is None or value != value:
        return None
    return float(value)


def start_wandb_run(
    *,
    mode: str,
    project: str,
    entity: str | None,
    name: str,
    config: dict[str, Any],
    resume_id: str | None = None,
    resume: str | None = None,
) -> Any | None:
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'WandB tracking requires `pip install -e ".[tracking]"` first.'
        ) from exc
    kwargs: dict[str, Any] = {
        "project": project,
        "entity": entity,
        "name": name,
        "mode": mode,
        "config": config,
        "job_type": "rllib-smoke-training",
    }
    if resume_id is not None:
        kwargs["id"] = resume_id
    if resume is not None:
        kwargs["resume"] = resume
    return wandb.init(**kwargs)


def log_iteration_result(
    run: Any | None,
    iteration: int,
    result: dict[str, Any],
) -> None:
    if run is None:
        return
    run.log(
        {
            "rllib/iteration": iteration,
            "rllib/num_env_steps_sampled": result.get("num_env_steps_sampled", 0),
            "rllib/num_agent_steps_sampled": result.get("num_agent_steps_sampled", 0),
            "rllib/episode_reward_mean": metric_from_result(
                result,
                "episode_reward_mean",
            ),
            "rllib/episode_len_mean": metric_from_result(result, "episode_len_mean"),
        },
        step=iteration,
    )


def log_summary(run: Any | None, result: RLLibTrainingResult) -> None:
    if run is None:
        return
    run.log(
        {
            "rllib/final_num_env_steps_sampled": result.num_env_steps_sampled,
            "rllib/final_num_agent_steps_sampled": result.num_agent_steps_sampled,
            "rllib/final_episode_reward_mean": result.episode_reward_mean,
            "rllib/final_episode_len_mean": result.episode_len_mean,
            "rllib/iterations": result.iterations,
        }
    )


class ConsoleTrainingProgress:
    def __init__(
        self,
        *,
        enabled: bool,
        total_iterations: int,
        run_name: str,
    ) -> None:
        self.enabled = enabled
        self.total_iterations = total_iterations
        self.run_name = run_name
        self.started_at = time.perf_counter()
        self._bar: Any | None = None

    def start(
        self,
        *,
        players: int,
        train_batch_size: int,
        evaluation_interval: int | None,
    ) -> None:
        if not self.enabled:
            return
        evaluation_text = (
            "off"
            if evaluation_interval is None
            else f"every {evaluation_interval} iteration(s)"
        )
        header = (
            "RLlib training "
            f"run={self.run_name} iterations={self.total_iterations} "
            f"players={players} train_batch_size={train_batch_size} "
            f"evaluation={evaluation_text}"
        )
        print(header)
        if tqdm is None:
            print("tqdm is not installed; showing evaluation summaries only.")
            return
        self._bar = tqdm(
            total=self.total_iterations,
            desc="training",
            unit="iter",
            dynamic_ncols=True,
            leave=True,
        )

    def set_phase(self, text: str) -> None:
        if not self.enabled or self._bar is None:
            return
        self._bar.set_description_str(text)

    def message(self, text: str) -> None:
        if not self.enabled:
            return
        if self._bar is not None:
            self._bar.write(text)
            return
        print(text)

    def update(self, iteration: int, result: dict[str, Any]) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started_at
        percent = iteration / self.total_iterations
        remaining_iterations = self.total_iterations - iteration
        seconds_per_iteration = elapsed / max(iteration, 1)
        eta_seconds = seconds_per_iteration * remaining_iterations
        env_steps = int(result.get("num_env_steps_sampled", 0))
        agent_steps = int(result.get("num_agent_steps_sampled", 0))
        steps_per_second = env_steps / elapsed if elapsed > 0 else 0.0
        reward = metric_from_result(result, "episode_reward_mean")
        episode_length = metric_from_result(result, "episode_len_mean")
        if self._bar is None:
            return
        delta = iteration - self._bar.n
        if delta > 0:
            self._bar.update(delta)
        self._bar.set_postfix(
            {
                "elapsed": format_duration(elapsed),
                "eta": format_duration(eta_seconds),
                "env/s": f"{steps_per_second:.1f}",
                "reward": format_optional_float(reward),
                "len": format_optional_float(episode_length),
                "env_steps": env_steps,
                "agent_steps": agent_steps,
            },
            refresh=True,
        )

    def evaluation_summary(
        self,
        iteration: int,
        evaluation_results: dict[str, Any],
    ) -> None:
        self.message(
            format_training_evaluation_status(
                iteration=iteration,
                total_iterations=self.total_iterations,
                started_at=self.started_at,
                evaluation_results=evaluation_results,
            )
        )

    def finish(self, result: RLLibTrainingResult) -> None:
        if not self.enabled:
            return
        self.close()
        elapsed = time.perf_counter() - self.started_at
        print(
            "RLlib training complete "
            f"elapsed={format_duration(elapsed)} "
            f"checkpoint={result.checkpoint_path}"
        )

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def progress_bar(fraction: float, *, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, fraction))
    filled = int(round(width * clamped))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def format_training_evaluation_status(
    *,
    iteration: int,
    total_iterations: int,
    started_at: float,
    evaluation_results: dict[str, Any],
) -> str:
    elapsed = time.perf_counter() - started_at
    remaining_iterations = max(0, total_iterations - iteration)
    seconds_per_iteration = elapsed / max(iteration, 1)
    eta_seconds = seconds_per_iteration * remaining_iterations
    return (
        f"iteration {iteration}/{total_iterations} "
        f"elapsed={format_duration(elapsed)} "
        f"eta={format_duration(eta_seconds)} "
        f"win_rates {format_evaluation_win_rates(evaluation_results)}"
    )


def format_evaluation_win_rates(evaluation_results: dict[str, Any]) -> str:
    if not evaluation_results:
        return "n/a"
    if is_randomized_seat_rank_evaluation_results(evaluation_results):
        return " ".join(
            (
                f"{policy_id}=win:{format_optional_float(result.win_rate)}"
                f"/rank:{format_optional_float(result.average_rank_reward)}"
            )
            for policy_id, result in evaluation_results.items()
        )
    if is_native_seat_evaluation_results(evaluation_results):
        return " | ".join(
            (
                f"{opponent}: "
                + " ".join(
                    (
                        f"{format_short_seat_name(seat)}="
                        f"{format_optional_float(result.rllib_combined_win_rate)}"
                    )
                    for seat, result in seat_results.items()
                )
            )
            for opponent, seat_results in evaluation_results.items()
        )
    return " ".join(
        (
            f"{opponent}="
            f"{format_optional_float(result.rllib_combined_win_rate)}"
        )
        for opponent, result in evaluation_results.items()
    )


def is_native_seat_evaluation_results(evaluation_results: dict[str, Any]) -> bool:
    return any(isinstance(value, dict) for value in evaluation_results.values())


def is_randomized_seat_rank_evaluation_results(
    evaluation_results: dict[str, Any],
) -> bool:
    return any(
        hasattr(value, "average_rank_reward")
        and hasattr(value, "average_finish_position")
        and hasattr(value, "place_rates")
        and hasattr(value, "seat_win_rates")
        for value in evaluation_results.values()
    )


def format_short_seat_name(seat: str) -> str:
    return seat.replace("player_", "p")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_optional_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def configure_rllib_console_noise(suppress_ray_warnings: bool) -> None:
    if not suppress_ray_warnings:
        return
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    os.environ.setdefault("RAY_DEDUP_LOGS", "1")
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="Tip: In future versions of Ray.*",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="overflow encountered in reduce",
        category=RuntimeWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="`_get_slice_indices` has been deprecated.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"var\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
    )


def import_ray() -> Any:
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError(
            'RLlib smoke training requires `pip install -e ".[multiagent]"` first.'
        ) from exc
    return ray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny shared-policy RLlib PPO smoke training job."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--players", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=None)
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=None)
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--hide-opponent-card-counts", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--rollout-fragment-length", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num-env-runners", type=int, default=None)
    parser.add_argument("--model-hidden-layers", type=parse_int_list, default=None)
    parser.add_argument("--model-activation", choices=MODEL_ACTIVATIONS, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--evaluation-interval", type=int, default=None)
    parser.add_argument("--evaluation-episodes", type=int, default=None)
    parser.add_argument("--evaluation-opponents", type=parse_name_list, default=None)
    parser.add_argument("--evaluation-scope", choices=EVALUATION_SCOPES, default=None)
    parser.add_argument("--restore-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--existing-run-action",
        choices=EXISTING_RUN_ACTIONS,
        default=None,
    )
    parser.add_argument("--continue-mode", choices=CONTINUE_MODES, default=None)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--show-ray-warnings", action="store_true")
    parser.add_argument("--dashboard-logging", action="store_true")
    parser.add_argument("--no-dashboard-logging", action="store_true")
    parser.add_argument("--experiment-dir", type=Path, default=None)
    parser.add_argument("--dashboard-run-name", default=None)
    return parser.parse_args()


def parse_optional_name_list(value: str) -> tuple[str, ...] | None:
    names = parse_name_list(value)
    return names or None


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def reveal_opponent_card_counts_override(args: argparse.Namespace) -> bool | None:
    if args.reveal_opponent_card_counts and args.hide_opponent_card_counts:
        raise ValueError(
            "Use only one of --reveal-opponent-card-counts or "
            "--hide-opponent-card-counts."
        )
    if args.reveal_opponent_card_counts:
        return True
    if args.hide_opponent_card_counts:
        return False
    return None


def show_progress_override(args: argparse.Namespace) -> bool | None:
    return False if args.no_progress else None


def suppress_ray_warnings_override(args: argparse.Namespace) -> bool | None:
    return False if args.show_ray_warnings else None


def dashboard_logging_enabled_override(args: argparse.Namespace) -> bool | None:
    if args.dashboard_logging and args.no_dashboard_logging:
        raise ValueError(
            "Use only one of --dashboard-logging or --no-dashboard-logging."
        )
    if args.dashboard_logging:
        return True
    if args.no_dashboard_logging:
        return False
    return None


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    config = apply_cli_overrides(config, args)
    config = prepare_config_for_existing_run(config)
    if config is None:
        return
    result = train_from_config(config)

    print("RLlib Shared-Policy Smoke Training")
    print(f"Iterations: {result.iterations}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Env steps sampled: {result.num_env_steps_sampled}")
    print(f"Agent steps sampled: {result.num_agent_steps_sampled}")
    print(f"Episode reward mean: {result.episode_reward_mean}")
    print(f"Episode length mean: {result.episode_len_mean}")


if __name__ == "__main__":
    main()
