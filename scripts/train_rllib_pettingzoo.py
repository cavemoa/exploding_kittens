from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from pettingzoo.utils import BaseWrapper

from exploding_kittens import pettingzoo_env


WANDB_MODES = ("online", "offline", "disabled")
DEFAULT_ENV_NAME = "exploding_kittens_rllib"
DEFAULT_MODEL_NAME = "exploding_kittens_action_mask_model"
SHARED_POLICY_ID = "shared_policy"


@dataclass(frozen=True)
class RLLibTrainingResult:
    iterations: int
    checkpoint_path: Path
    num_env_steps_sampled: int
    num_agent_steps_sampled: int
    episode_reward_mean: float | None
    episode_len_mean: float | None


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
    checkpoint_dir: Path = Path("models/rllib"),
    run_name: str = "rllib_shared_policy_smoke",
    train_batch_size: int = 64,
    minibatch_size: int = 32,
    rollout_fragment_length: int = 32,
    num_epochs: int = 1,
    lr: float = 0.0003,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> RLLibTrainingResult:
    validate_training_args(
        iterations=iterations,
        players=players,
        train_batch_size=train_batch_size,
        minibatch_size=minibatch_size,
        rollout_fragment_length=rollout_fragment_length,
        wandb_mode=wandb_mode,
    )
    config_payload = {
        "iterations": iterations,
        "players": players,
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "run_name": run_name,
        "algorithm": "RLlib PPO",
        "policy_mapping": "shared_policy",
        "train_batch_size": train_batch_size,
        "minibatch_size": minibatch_size,
        "rollout_fragment_length": rollout_fragment_length,
        "num_epochs": num_epochs,
        "lr": lr,
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config_payload,
    )

    ray = import_ray()
    algorithm = None
    ray_started_here = not ray.is_initialized()
    if ray_started_here:
        ray.init(include_dashboard=False, ignore_reinit_error=True, num_cpus=1)

    try:
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
        )
        register_action_mask_model(DEFAULT_MODEL_NAME)
        config = build_ppo_config(
            env_name=env_name,
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            rollout_fragment_length=rollout_fragment_length,
            num_epochs=num_epochs,
            lr=lr,
            model_name=DEFAULT_MODEL_NAME,
        )
        algorithm = config.build_algo()
        last_result: dict[str, Any] = {}
        for iteration in range(1, iterations + 1):
            last_result = algorithm.train()
            log_iteration_result(run, iteration, last_result)

        checkpoint_root = checkpoint_dir / run_name
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_root.resolve()
        algorithm.save(str(checkpoint_path))
        result = RLLibTrainingResult(
            iterations=iterations,
            checkpoint_path=checkpoint_path,
            num_env_steps_sampled=int(last_result.get("num_env_steps_sampled", 0)),
            num_agent_steps_sampled=int(last_result.get("num_agent_steps_sampled", 0)),
            episode_reward_mean=metric_from_result(last_result, "episode_reward_mean"),
            episode_len_mean=metric_from_result(last_result, "episode_len_mean"),
        )
        log_summary(run, result)
        return result
    finally:
        if algorithm is not None:
            algorithm.stop()
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
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


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
    model_name: str = DEFAULT_MODEL_NAME,
) -> Any:
    from ray.rllib.algorithms.ppo import PPOConfig

    return (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env_name, disable_env_checking=True)
        .framework("torch")
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length=rollout_fragment_length,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            model={"custom_model": model_name},
        )
        .multi_agent(
            policies={SHARED_POLICY_ID},
            policy_mapping_fn=shared_policy_mapping,
        )
        .resources(num_gpus=0)
    )


def shared_policy_mapping(agent_id: str, episode: Any, **kwargs: Any) -> str:
    del agent_id, episode, kwargs
    return SHARED_POLICY_ID


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
) -> Any | None:
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'WandB tracking requires `pip install -e ".[tracking]"` first.'
        ) from exc
    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        config=config,
        job_type="rllib-smoke-training",
    )


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
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/rllib"))
    parser.add_argument("--run-name", default="rllib_shared_policy_smoke")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--rollout-fragment-length", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    parser.add_argument("--wandb-project", default="exploding-kittens")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def parse_optional_name_list(value: str) -> tuple[str, ...] | None:
    names = parse_name_list(value)
    return names or None


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main() -> None:
    args = parse_args()
    result = train_rllib_shared_policy(
        iterations=args.iterations,
        players=args.players,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        train_batch_size=args.train_batch_size,
        minibatch_size=args.minibatch_size,
        rollout_fragment_length=args.rollout_fragment_length,
        num_epochs=args.num_epochs,
        lr=args.lr,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("RLlib Shared-Policy Smoke Training")
    print(f"Iterations: {result.iterations}")
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Env steps sampled: {result.num_env_steps_sampled}")
    print(f"Agent steps sampled: {result.num_agent_steps_sampled}")
    print(f"Episode reward mean: {result.episode_reward_mean}")
    print(f"Episode length mean: {result.episode_len_mean}")


if __name__ == "__main__":
    main()
