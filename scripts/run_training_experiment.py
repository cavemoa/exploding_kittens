from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_agent import SavedPolicyEvaluationResult, evaluate_saved_policy
    from scripts.evaluate_single_agent import EvaluationResult, evaluate_single_agent
    from scripts.train_single_agent import TrainingResult, train_single_agent
except ModuleNotFoundError:
    from evaluate_agent import SavedPolicyEvaluationResult, evaluate_saved_policy
    from evaluate_single_agent import EvaluationResult, evaluate_single_agent
    from train_single_agent import TrainingResult, train_single_agent


WANDB_MODES = ("online", "offline", "disabled")
DEFAULT_OPPONENT_STRATEGIES = ("draw-only", "random", "safe-rule")


@dataclass(frozen=True)
class ExperimentComparisonRow:
    opponent_strategy: str
    trained_win_rate: float
    random_win_rate: float
    win_rate_delta: float
    trained_average_reward: float
    random_average_reward: float
    reward_delta: float
    trained_average_turns: float
    random_average_turns: float


@dataclass(frozen=True)
class VisibleExperimentResult:
    run_name: str
    model_path: Path
    report_path: Path
    training: TrainingResult
    comparison_rows: tuple[ExperimentComparisonRow, ...]


def run_training_experiment(
    *,
    total_timesteps: int = 5_000,
    episodes: int = 100,
    players: int = 2,
    learner: str = "player_1",
    training_opponent_strategy: str = "draw-only",
    opponent_strategies: tuple[str, ...] = DEFAULT_OPPONENT_STRATEGIES,
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    model_dir: Path = Path("models"),
    report_dir: Path = Path("reports"),
    run_name: str = "visible_training_experiment",
    n_steps: int = 64,
    batch_size: int = 32,
    n_epochs: int = 2,
    learning_rate: float = 0.0003,
    training_eval_episodes: int = 5,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> VisibleExperimentResult:
    validate_experiment_args(
        total_timesteps=total_timesteps,
        episodes=episodes,
        opponent_strategies=opponent_strategies,
        wandb_mode=wandb_mode,
    )
    config = {
        "total_timesteps": total_timesteps,
        "episodes": episodes,
        "players": players,
        "learner": learner,
        "training_opponent_strategy": training_opponent_strategy,
        "opponent_strategies": opponent_strategies,
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "run_name": run_name,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "training_eval_episodes": training_eval_episodes,
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=config,
    )
    try:
        training_result = train_single_agent(
            total_timesteps=total_timesteps,
            players=players,
            learner=learner,
            opponent_strategy=training_opponent_strategy,
            seed=seed,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            model_dir=model_dir,
            run_name=run_name,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            eval_episodes=training_eval_episodes,
            wandb_mode="disabled",
        )
        trained_result = evaluate_saved_policy(
            model_path=training_result.model_path,
            episodes=episodes,
            players=players,
            learner=learner,
            opponent_strategies=opponent_strategies,
            seed=seed + 20_000,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            wandb_mode="disabled",
        )
        random_results = {
            opponent_strategy: evaluate_single_agent(
                episodes=episodes,
                players=players,
                learner=learner,
                opponent_strategy=opponent_strategy,
                seed=seed + 30_000 + strategy_index * 10_000,
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                trace_episodes=0,
                wandb_mode="disabled",
            )
            for strategy_index, opponent_strategy in enumerate(opponent_strategies)
        }
        comparison_rows = build_comparison_rows(trained_result, random_results)
        report_path = write_experiment_report(
            report_dir=report_dir,
            run_name=run_name,
            seed=seed,
            config=config,
            training_result=training_result,
            trained_result=trained_result,
            random_results=random_results,
            comparison_rows=comparison_rows,
        )
        result = VisibleExperimentResult(
            run_name=run_name,
            model_path=training_result.model_path,
            report_path=report_path,
            training=training_result,
            comparison_rows=comparison_rows,
        )
        log_experiment_results(run, result)
        return result
    finally:
        if run is not None:
            run.finish()


def validate_experiment_args(
    *,
    total_timesteps: int,
    episodes: int,
    opponent_strategies: tuple[str, ...],
    wandb_mode: str,
) -> None:
    if total_timesteps < 1:
        raise ValueError("total_timesteps must be at least 1.")
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if not opponent_strategies:
        raise ValueError("At least one opponent strategy is required.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


def build_comparison_rows(
    trained_result: SavedPolicyEvaluationResult,
    random_results: dict[str, EvaluationResult],
) -> tuple[ExperimentComparisonRow, ...]:
    rows: list[ExperimentComparisonRow] = []
    for opponent_result in trained_result.opponent_results:
        random_result = random_results[opponent_result.opponent_strategy]
        rows.append(
            ExperimentComparisonRow(
                opponent_strategy=opponent_result.opponent_strategy,
                trained_win_rate=opponent_result.win_rate,
                random_win_rate=random_result.win_rate,
                win_rate_delta=opponent_result.win_rate - random_result.win_rate,
                trained_average_reward=opponent_result.average_reward,
                random_average_reward=random_result.average_reward,
                reward_delta=opponent_result.average_reward - random_result.average_reward,
                trained_average_turns=opponent_result.average_turns_survived,
                random_average_turns=random_result.average_turns,
            )
        )
    return tuple(rows)


def write_experiment_report(
    *,
    report_dir: Path,
    run_name: str,
    seed: int,
    config: dict[str, Any],
    training_result: TrainingResult,
    trained_result: SavedPolicyEvaluationResult,
    random_results: dict[str, EvaluationResult],
    comparison_rows: tuple[ExperimentComparisonRow, ...],
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{run_name}_seed_{seed}.json"
    payload = {
        "config": make_jsonable(config),
        "training": make_jsonable(asdict(training_result)),
        "trained_policy": make_jsonable(asdict(trained_result)),
        "random_masked_baselines": {
            opponent_strategy: make_jsonable(asdict(result))
            for opponent_strategy, result in random_results.items()
        },
        "comparison": [asdict(row) for row in comparison_rows],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [make_jsonable(item) for item in value]
    if isinstance(value, list):
        return [make_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_jsonable(item) for key, item in value.items()}
    return value


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
        job_type="visible-training-experiment",
    )


def log_experiment_results(
    run: Any | None,
    result: VisibleExperimentResult,
) -> None:
    if run is None:
        return

    for row in result.comparison_rows:
        prefix = f"comparison/{row.opponent_strategy}"
        run.log(
            {
                f"{prefix}/trained_win_rate": row.trained_win_rate,
                f"{prefix}/random_win_rate": row.random_win_rate,
                f"{prefix}/win_rate_delta": row.win_rate_delta,
                f"{prefix}/trained_average_reward": row.trained_average_reward,
                f"{prefix}/random_average_reward": row.random_average_reward,
                f"{prefix}/reward_delta": row.reward_delta,
            }
        )
    log_comparison_table(run, result)


def log_comparison_table(
    run: Any | None,
    result: VisibleExperimentResult,
) -> None:
    if run is None:
        return

    import wandb

    table = wandb.Table(
        columns=[
            "opponent_strategy",
            "trained_win_rate",
            "random_win_rate",
            "win_rate_delta",
            "trained_average_reward",
            "random_average_reward",
            "reward_delta",
            "trained_average_turns",
            "random_average_turns",
        ]
    )
    for row in result.comparison_rows:
        table.add_data(
            row.opponent_strategy,
            row.trained_win_rate,
            row.random_win_rate,
            row.win_rate_delta,
            row.trained_average_reward,
            row.random_average_reward,
            row.reward_delta,
            row.trained_average_turns,
            row.random_average_turns,
        )
    run.log({"comparison/trained_vs_random": table})


def format_comparison_table(rows: tuple[ExperimentComparisonRow, ...]) -> str:
    headers = (
        "opponent",
        "trained_win",
        "random_win",
        "delta",
        "trained_reward",
        "random_reward",
    )
    values = [
        (
            row.opponent_strategy,
            f"{row.trained_win_rate:.3f}",
            f"{row.random_win_rate:.3f}",
            f"{row.win_rate_delta:+.3f}",
            f"{row.trained_average_reward:.3f}",
            f"{row.random_average_reward:.3f}",
        )
        for row in rows
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in values))
        for column in range(len(headers))
    ]
    separator = "+".join("-" * (width + 2) for width in widths)
    lines = [
        separator,
        "|".join(
            f" {headers[column]:<{widths[column]}} "
            for column in range(len(headers))
        ),
        separator,
    ]
    for row in values:
        lines.append(
            "|".join(
                f" {row[column]:<{widths[column]}} "
                for column in range(len(headers))
            )
        )
    lines.append(separator)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train once, evaluate against baselines, and print/log a comparison."
    )
    parser.add_argument("--total-timesteps", type=int, default=5_000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--learner", default="player_1")
    parser.add_argument("--training-opponent-strategy", default="draw-only")
    parser.add_argument(
        "--opponent-strategies",
        type=parse_name_list,
        default=DEFAULT_OPPONENT_STRATEGIES,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--run-name", default="visible_training_experiment")
    parser.add_argument("--n-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--training-eval-episodes", type=int, default=5)
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
    result = run_training_experiment(
        total_timesteps=args.total_timesteps,
        episodes=args.episodes,
        players=args.players,
        learner=args.learner,
        training_opponent_strategy=args.training_opponent_strategy,
        opponent_strategies=args.opponent_strategies,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        model_dir=args.model_dir,
        report_dir=args.report_dir,
        run_name=args.run_name,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        training_eval_episodes=args.training_eval_episodes,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("Visible Training Experiment")
    print(f"Run: {result.run_name}")
    print(f"Model: {result.model_path}")
    print(f"Report: {result.report_path}")
    print(format_comparison_table(result.comparison_rows))


if __name__ == "__main__":
    main()
