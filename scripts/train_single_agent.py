from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exploding_kittens import SingleAgentEnv


WANDB_MODES = ("online", "offline", "disabled")


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    total_timesteps: int
    eval_episodes: int
    eval_wins: int
    eval_win_rate: float
    eval_average_reward: float
    eval_average_turns: float


def train_single_agent(
    *,
    total_timesteps: int = 64,
    players: int = 2,
    learner: str = "player_1",
    opponent_strategy: str = "draw-only",
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    model_dir: Path = Path("models"),
    run_name: str = "maskable_ppo_single_agent",
    n_steps: int = 16,
    batch_size: int = 8,
    n_epochs: int = 1,
    learning_rate: float = 0.0003,
    eval_episodes: int = 5,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> TrainingResult:
    validate_training_args(
        total_timesteps=total_timesteps,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        eval_episodes=eval_episodes,
        wandb_mode=wandb_mode,
    )
    config = {
        "total_timesteps": total_timesteps,
        "players": players,
        "learner": learner,
        "opponent_strategy": opponent_strategy,
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "algorithm": "MaskablePPO",
        "policy": "MultiInputPolicy",
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "eval_episodes": eval_episodes,
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{run_name}_seed_{seed}.zip"

    environment = make_training_env(
        players=players,
        learner=learner,
        opponent_strategy=opponent_strategy,
        seed=seed,
        max_turns=max_turns,
        include_cards=include_cards,
        exclude_cards=exclude_cards,
        enabled_combo_rules=enabled_combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )
    model = make_model(
        environment,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
    )

    try:
        callback = make_wandb_callback(run, model_dir=model_dir, model_save_freq=n_steps)
        model.learn(total_timesteps=total_timesteps, callback=callback)
        model.save(model_path)
        eval_metrics = evaluate_model(
            model,
            episodes=eval_episodes,
            players=players,
            learner=learner,
            opponent_strategy=opponent_strategy,
            seed=seed + 10_000,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
        )
        result = TrainingResult(
            model_path=model_path,
            total_timesteps=total_timesteps,
            eval_episodes=eval_episodes,
            eval_wins=eval_metrics["wins"],
            eval_win_rate=eval_metrics["win_rate"],
            eval_average_reward=eval_metrics["average_reward"],
            eval_average_turns=eval_metrics["average_turns"],
        )
        log_training_summary(run, result)
        return result
    finally:
        environment.close()
        if run is not None:
            run.finish()


def validate_training_args(
    *,
    total_timesteps: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    eval_episodes: int,
    wandb_mode: str,
) -> None:
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1.")
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2 for PPO rollouts.")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2.")
    if batch_size > n_steps:
        raise ValueError("batch_size should not exceed n_steps for the tiny single-env run.")
    if n_epochs < 1:
        raise ValueError("n_epochs must be at least 1.")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


def make_training_env(
    *,
    players: int,
    learner: str,
    opponent_strategy: str,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
) -> Any:
    try:
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            'Mask-aware training requires `pip install -e ".[training]"` first.'
        ) from exc

    environment = SingleAgentEnv(
        players=players,
        learner=learner,
        opponent_strategy=opponent_strategy,
        seed=seed,
        max_turns=max_turns,
        include_cards=include_cards,
        exclude_cards=exclude_cards,
        enabled_combo_rules=enabled_combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )
    return Monitor(environment)


def make_model(
    environment: Any,
    *,
    seed: int,
    n_steps: int,
    batch_size: int,
    n_epochs: int,
    learning_rate: float,
) -> Any:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import is_masking_supported

    if not is_masking_supported(environment):
        raise RuntimeError("Training environment does not expose action_masks().")

    return MaskablePPO(
        "MultiInputPolicy",
        environment,
        seed=seed,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        verbose=0,
    )


def evaluate_model(
    model: Any,
    *,
    episodes: int,
    players: int,
    learner: str,
    opponent_strategy: str,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
) -> dict[str, float | int]:
    rewards: list[float] = []
    turns: list[int] = []
    wins = 0
    for episode_index in range(episodes):
        environment = SingleAgentEnv(
            players=players,
            learner=learner,
            opponent_strategy=opponent_strategy,
            seed=seed + episode_index,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
        )
        observation, _ = environment.reset(seed=seed + episode_index)
        terminated = False
        truncated = False
        total_reward = 0.0
        info: dict[str, Any] = {"winner": None, "turn_count": 0}
        while not terminated and not truncated:
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=environment.action_masks(),
            )
            observation, reward, terminated, truncated, info = environment.step(int(action))
            total_reward += reward

        rewards.append(total_reward)
        turns.append(info["turn_count"])
        wins += int(info["winner"] == learner)
        environment.close()

    return {
        "wins": wins,
        "win_rate": wins / episodes,
        "average_reward": float(np.mean(rewards)),
        "average_turns": float(np.mean(turns)),
    }


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
        job_type="training",
    )


def make_wandb_callback(
    run: Any | None,
    *,
    model_dir: Path,
    model_save_freq: int,
) -> Any | None:
    if run is None:
        return None

    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from wandb.integration.sb3 import WandbCallback

    class WandbEpisodeCallback(BaseCallback):
        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is None:
                    continue
                run.log(
                    {
                        "train/episode_reward": episode["r"],
                        "train/episode_length": episode["l"],
                    },
                    step=self.num_timesteps,
                )
            return True

    checkpoint_callback = WandbCallback(
        model_save_path=str(model_dir / "wandb_checkpoints"),
        model_save_freq=model_save_freq,
        gradient_save_freq=0,
        log=None,
    )
    return CallbackList([WandbEpisodeCallback(), checkpoint_callback])


def log_training_summary(run: Any | None, result: TrainingResult) -> None:
    if run is None:
        return
    run.log(
        {
            "eval/win_rate": result.eval_win_rate,
            "eval/average_reward": result.eval_average_reward,
            "eval/average_turns": result.eval_average_turns,
            "eval/wins": result.eval_wins,
            "training/total_timesteps": result.total_timesteps,
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single learner with MaskablePPO against scripted opponents."
    )
    parser.add_argument("--total-timesteps", type=int, default=64)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--learner", default="player_1")
    parser.add_argument("--opponent-strategy", default="draw-only")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--run-name", default="maskable_ppo_single_agent")
    parser.add_argument("--n-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--eval-episodes", type=int, default=5)
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
    result = train_single_agent(
        total_timesteps=args.total_timesteps,
        players=args.players,
        learner=args.learner,
        opponent_strategy=args.opponent_strategy,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        model_dir=args.model_dir,
        run_name=args.run_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        eval_episodes=args.eval_episodes,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("Mask-Aware PPO Training")
    print(f"Model: {result.model_path}")
    print(f"Total timesteps: {result.total_timesteps}")
    print(f"Evaluation episodes: {result.eval_episodes}")
    print(f"Evaluation wins: {result.eval_wins}")
    print(f"Evaluation win rate: {result.eval_win_rate:.3f}")
    print(f"Evaluation average reward: {result.eval_average_reward:.3f}")
    print(f"Evaluation average turns: {result.eval_average_turns:.1f}")


if __name__ == "__main__":
    main()
