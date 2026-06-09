from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploding_kittens.experiment_logging import (  # noqa: E402
    ExperimentRunData,
    discover_experiment_runs,
)


DEFAULT_EXPERIMENT_DIR = Path("reports/experiments")
DEFAULT_REFRESH_SECONDS = 5
STALE_RUN_SECONDS = 120
DEFAULT_CONFIG_DIR = Path("configs")
ACTIVE_LAUNCH_REFRESH_SECONDS = 1
MAX_LAUNCHER_LOG_LINES = 500
WANDB_MODE_OPTIONS = ("use-config", "disabled", "offline", "online")


@dataclass(frozen=True)
class DashboardRunSummary:
    run_name: str
    run_dir: Path
    status: str
    current_iteration: int | None
    total_iterations: int | None
    latest_reward_mean: float | None
    latest_episode_length_mean: float | None
    latest_environment_steps_per_second: float | None
    latest_win_rates: dict[str, float]
    elapsed_seconds: float | None
    estimated_seconds_remaining: float | None
    latest_checkpoint_path: str | None
    active_config_name: str | None
    last_updated_at: str | None
    is_stale: bool
    metrics: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    checkpoints: tuple[dict[str, Any], ...]
    resolved_config_text: str | None


def load_dashboard_runs(
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
) -> tuple[DashboardRunSummary, ...]:
    runs = tuple(summarize_run(run) for run in discover_experiment_runs(experiment_dir))
    return tuple(sorted(runs, key=dashboard_run_sort_key, reverse=True))


def summarize_run(run: ExperimentRunData) -> DashboardRunSummary:
    summary = run.summary
    latest_metric = run.metrics[-1] if run.metrics else {}
    latest_checkpoint = run.checkpoints[-1] if run.checkpoints else {}
    last_updated_at = latest_activity_timestamp(run)
    return DashboardRunSummary(
        run_name=str(summary.get("run_name") or run.run_dir.name),
        run_dir=run.run_dir,
        status=str(summary.get("status") or "unknown"),
        current_iteration=optional_int(latest_metric.get("iteration")),
        total_iterations=optional_int(latest_metric.get("total_iterations")),
        latest_reward_mean=optional_float(latest_metric.get("episode_reward_mean")),
        latest_episode_length_mean=optional_float(
            latest_metric.get("episode_length_mean")
        ),
        latest_environment_steps_per_second=optional_float(
            latest_metric.get("environment_steps_per_second")
        ),
        latest_win_rates=latest_win_rates(run.evaluations),
        elapsed_seconds=optional_float(latest_metric.get("elapsed_seconds")),
        estimated_seconds_remaining=optional_float(
            latest_metric.get("estimated_seconds_remaining")
        ),
        latest_checkpoint_path=latest_checkpoint_path(summary, latest_checkpoint),
        active_config_name=(
            "resolved_config.yaml" if run.resolved_config_text is not None else None
        ),
        last_updated_at=last_updated_at,
        is_stale=is_stale_run(
            status=str(summary.get("status") or "unknown"),
            last_updated_at=last_updated_at,
        ),
        metrics=run.metrics,
        evaluations=run.evaluations,
        checkpoints=run.checkpoints,
        resolved_config_text=run.resolved_config_text,
    )


def latest_win_rates(evaluations: tuple[dict[str, Any], ...]) -> dict[str, float]:
    by_opponent: dict[str, float] = {}
    for row in evaluations:
        opponent = row.get("opponent_strategy")
        win_rate = optional_float(row.get("rllib_combined_win_rate"))
        if opponent is not None and win_rate is not None:
            by_opponent[str(opponent)] = win_rate
    return by_opponent


def dashboard_run_sort_key(run: DashboardRunSummary) -> tuple[str, str]:
    return (run.last_updated_at or "", run.run_name)


def latest_activity_timestamp(run: ExperimentRunData) -> str | None:
    candidates = [
        run.summary.get("last_updated_at"),
        *(row.get("logged_at") for row in run.metrics),
        *(row.get("logged_at") for row in run.evaluations),
        *(row.get("logged_at") for row in run.checkpoints),
    ]
    timestamps = [str(value) for value in candidates if value]
    return max(timestamps) if timestamps else None


def latest_checkpoint_path(
    summary: dict[str, Any],
    latest_checkpoint: dict[str, Any],
) -> str | None:
    checkpoint_path = summary.get("final_checkpoint_path") or latest_checkpoint.get(
        "checkpoint_path"
    )
    return str(checkpoint_path) if checkpoint_path else None


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def is_stale_run(
    *,
    status: str,
    last_updated_at: str | None,
    stale_after_seconds: int = STALE_RUN_SECONDS,
    now: datetime | None = None,
) -> bool:
    if status != "running" or last_updated_at is None:
        return False
    current_time = now or datetime.now(UTC)
    try:
        updated_at = datetime.fromisoformat(last_updated_at)
    except ValueError:
        return True
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    return (current_time - updated_at).total_seconds() > stale_after_seconds


def format_progress(run: DashboardRunSummary) -> str:
    if run.current_iteration is None or run.total_iterations is None:
        return "No iteration data yet"
    if run.total_iterations <= 0:
        return f"{run.current_iteration} iterations"
    percentage = run.current_iteration / run.total_iterations * 100
    return f"{run.current_iteration}/{run.total_iterations} ({percentage:.1f}%)"


def format_metric(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def format_duration_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    total_seconds = max(0, int(round(value)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def metric_points(
    metrics: tuple[dict[str, Any], ...],
    metric_name: str,
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for row in metrics:
        iteration = optional_float(row.get("iteration"))
        value = optional_float(row.get(metric_name))
        if iteration is not None and value is not None:
            points.append((iteration, value))
    return tuple(points)


def smooth_points(
    points: tuple[tuple[float, float], ...],
    *,
    window: int = 3,
) -> tuple[tuple[float, float], ...]:
    if window <= 1 or len(points) <= 1:
        return points
    smoothed: list[tuple[float, float]] = []
    for index, (x_value, _) in enumerate(points):
        window_points = points[max(0, index - window + 1) : index + 1]
        mean_value = sum(value for _, value in window_points) / len(window_points)
        smoothed.append((x_value, mean_value))
    return tuple(smoothed)


def chart_bounds(
    points: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 1.0, 0.0, 1.0)
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    min_x = min(x_values)
    max_x = max(x_values)
    min_y = min(y_values)
    max_y = max(y_values)
    if min_x == max_x:
        max_x = min_x + 1.0
    if min_y == max_y:
        padding = max(abs(min_y) * 0.1, 1.0)
    else:
        padding = (max_y - min_y) * 0.1
    return (min_x, max_x, min_y - padding, max_y + padding)


def axis_label_values(min_value: float, max_value: float) -> tuple[float, ...]:
    if min_value == max_value:
        return (min_value,)
    midpoint = (min_value + max_value) / 2
    values = (min_value, midpoint, max_value)
    deduped: list[float] = []
    seen: set[str] = set()
    for value in values:
        label = format_axis_value(value)
        if label not in seen:
            deduped.append(value)
            seen.add(label)
    return tuple(deduped)


def format_axis_value(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if float(value).is_integer():
        return f"{value:.0f}"
    if abs(value) < 0.01:
        return f"{value:.3g}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def latest_training_metric_rows(
    metrics: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str], ...]:
    if not metrics:
        return ()
    latest = metrics[-1]
    return (
        ("Iteration", format_metric(optional_float(latest.get("iteration")), digits=0)),
        (
            "Reward Mean",
            format_metric(optional_float(latest.get("episode_reward_mean"))),
        ),
        (
            "Episode Length Mean",
            format_metric(optional_float(latest.get("episode_length_mean"))),
        ),
        (
            "Environment Steps",
            format_metric(optional_float(latest.get("environment_steps_sampled")), digits=0),
        ),
        (
            "Agent Steps",
            format_metric(optional_float(latest.get("agent_steps_sampled")), digits=0),
        ),
        (
            "Environment Steps/s",
            format_metric(
                optional_float(latest.get("environment_steps_per_second")),
                digits=1,
            ),
        ),
        (
            "Elapsed",
            format_duration_seconds(optional_float(latest.get("elapsed_seconds"))),
        ),
        (
            "ETA",
            format_duration_seconds(
                optional_float(latest.get("estimated_seconds_remaining"))
            ),
        ),
    )


def evaluation_points(
    evaluations: tuple[dict[str, Any], ...],
    opponent: str,
    metric_name: str,
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for row in evaluations:
        if row.get("opponent_strategy") != opponent:
            continue
        iteration = optional_float(row.get("iteration"))
        value = optional_float(row.get(metric_name))
        if iteration is not None and value is not None:
            points.append((iteration, value))
    return tuple(points)


def latest_evaluation_metric_rows(
    evaluations: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str, str, str, str, str, str], ...]:
    latest_by_opponent: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        opponent = row.get("opponent_strategy")
        if opponent is not None:
            latest_by_opponent[str(opponent)] = row
    rows: list[tuple[str, str, str, str, str, str, str]] = []
    for opponent in sorted(latest_by_opponent):
        row = latest_by_opponent[opponent]
        rows.append(
            (
                opponent,
                format_metric(optional_float(row.get("evaluation_episodes")), digits=0),
                format_metric(optional_float(row.get("rllib_combined_win_rate"))),
                format_metric(optional_float(row.get("scripted_opponent_win_rate"))),
                format_metric(optional_float(row.get("seat_order_advantage"))),
                format_metric(optional_float(row.get("illegal_action_rate"))),
                format_metric(optional_float(row.get("combo_count")), digits=0),
            )
        )
    return tuple(rows)


def latest_evaluations_by_opponent(
    evaluations: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    latest_by_opponent: dict[str, dict[str, Any]] = {}
    for row in evaluations:
        opponent = row.get("opponent_strategy")
        if opponent is not None:
            latest_by_opponent[str(opponent)] = row
    return latest_by_opponent


def best_evaluation_checkpoint(
    evaluations: tuple[dict[str, Any], ...],
    checkpoints: tuple[dict[str, Any], ...],
) -> tuple[int, float, str | None] | None:
    if not evaluations:
        return None
    scores_by_iteration: dict[int, list[float]] = {}
    for row in evaluations:
        iteration = optional_int(row.get("iteration"))
        score = optional_float(row.get("rllib_combined_win_rate"))
        if iteration is not None and score is not None:
            scores_by_iteration.setdefault(iteration, []).append(score)
    if not scores_by_iteration:
        return None
    best_iteration, scores = max(
        scores_by_iteration.items(),
        key=lambda item: (sum(item[1]) / len(item[1]), item[0]),
    )
    best_score = sum(scores) / len(scores)
    checkpoint_path = checkpoint_for_iteration(checkpoints, best_iteration)
    return best_iteration, best_score, checkpoint_path


def checkpoint_for_iteration(
    checkpoints: tuple[dict[str, Any], ...],
    iteration: int,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for row in checkpoints:
        checkpoint_iteration = optional_int(row.get("iteration"))
        checkpoint_path = row.get("checkpoint_path")
        if checkpoint_iteration is not None and checkpoint_path:
            candidates.append((checkpoint_iteration, str(checkpoint_path)))
    if not candidates:
        return None
    exact = [
        path
        for checkpoint_iteration, path in candidates
        if checkpoint_iteration == iteration
    ]
    if exact:
        return exact[-1]
    earlier = [
        (checkpoint_iteration, path)
        for checkpoint_iteration, path in candidates
        if checkpoint_iteration <= iteration
    ]
    if earlier:
        return max(earlier, key=lambda item: item[0])[1]
    return min(candidates, key=lambda item: item[0])[1]


def action_category_counts(
    evaluations: tuple[dict[str, Any], ...],
) -> dict[str, int]:
    counts = {
        "draw": 0,
        "play card": 0,
        "targeted card action": 0,
        "combo action": 0,
    }
    for row in latest_evaluations_by_opponent(evaluations).values():
        for action, count in dict(row.get("action_distribution") or {}).items():
            category = action_category(str(action))
            if category in counts:
                counts[category] += int(count)
    return counts


def action_category(action_name: str) -> str:
    if action_name == "draw":
        return "draw"
    if "two_of_a_kind" in action_name or "combo" in action_name:
        return "combo action"
    if action_name in {"play_favor"}:
        return "targeted card action"
    if action_name.startswith("play_"):
        return "play card"
    return action_name


def latest_card_distribution_rows(
    evaluations: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = []
    for opponent, row in sorted(latest_evaluations_by_opponent(evaluations).items()):
        card_distribution = dict(row.get("card_distribution") or {})
        for card, count in sorted(card_distribution.items()):
            rows.append((opponent, str(card), str(count)))
    return tuple(rows)


def latest_action_distribution_rows(
    evaluations: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str]] = []
    for opponent, row in sorted(latest_evaluations_by_opponent(evaluations).items()):
        action_distribution = dict(row.get("action_distribution") or {})
        for action, count in sorted(action_distribution.items()):
            rows.append((opponent, action_category(str(action)), str(action), str(count)))
    return tuple(rows)


def combo_rate_rows(
    evaluations: tuple[dict[str, Any], ...],
) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for opponent, row in sorted(latest_evaluations_by_opponent(evaluations).items()):
        combo_count = optional_float(row.get("combo_count")) or 0.0
        episodes = optional_float(row.get("evaluation_episodes")) or 0.0
        combo_rate = combo_count / episodes if episodes > 0 else None
        rows.append((opponent, format_metric(combo_rate)))
    return tuple(rows)


def suspicious_behavior_messages(
    evaluations: tuple[dict[str, Any], ...],
) -> tuple[str, ...]:
    latest_rows = latest_evaluations_by_opponent(evaluations)
    if not latest_rows:
        return ()
    action_counts = action_category_counts(evaluations)
    total_actions = sum(action_counts.values())
    messages: list[str] = []
    if total_actions > 0 and action_counts["draw"] / total_actions >= 0.8:
        messages.append("High draw rate: latest evaluations are mostly draw actions.")
    card_play_actions = (
        action_counts["play card"]
        + action_counts["targeted card action"]
        + action_counts["combo action"]
    )
    if total_actions > 0 and card_play_actions == 0:
        messages.append("No card-play actions detected in latest evaluations.")
    for opponent, row in sorted(latest_rows.items()):
        illegal_action_rate = optional_float(row.get("illegal_action_rate")) or 0.0
        if illegal_action_rate > 0:
            messages.append(
                f"Illegal action rate against {opponent}: {illegal_action_rate:.3f}."
            )
    return tuple(messages)


def evaluation_scores_by_iteration(
    evaluations: tuple[dict[str, Any], ...],
) -> dict[int, float]:
    scores_by_iteration: dict[int, list[float]] = {}
    for row in evaluations:
        iteration = optional_int(row.get("iteration"))
        score = optional_float(row.get("rllib_combined_win_rate"))
        if iteration is not None and score is not None:
            scores_by_iteration.setdefault(iteration, []).append(score)
    return {
        iteration: sum(scores) / len(scores)
        for iteration, scores in scores_by_iteration.items()
    }


def checkpoint_table_rows(
    checkpoints: tuple[dict[str, Any], ...],
    evaluations: tuple[dict[str, Any], ...],
    latest_checkpoint: str | None,
) -> tuple[tuple[str, str, str, str, str], ...]:
    scores = evaluation_scores_by_iteration(evaluations)
    scored_iterations = set(scores)
    if scores:
        best_iteration = max(scores, key=lambda iteration: (scores[iteration], iteration))
    else:
        best_iteration = None
    rows: list[tuple[str, str, str, str, str]] = []
    for row in sorted(
        checkpoints,
        key=lambda checkpoint: optional_int(checkpoint.get("iteration")) or -1,
    ):
        iteration = optional_int(row.get("iteration"))
        path = str(row.get("checkpoint_path") or "")
        explicit_score = optional_float(row.get("best_known_evaluation_score"))
        score = explicit_score
        if score is None and iteration in scored_iterations:
            score = scores[iteration or 0]
        markers: list[str] = []
        if latest_checkpoint and path == latest_checkpoint:
            markers.append("latest")
        if best_iteration is not None and iteration == best_iteration:
            markers.append("best")
        rows.append(
            (
                format_metric(float(iteration), digits=0) if iteration is not None else "n/a",
                str(row.get("created_at") or row.get("logged_at") or "n/a"),
                format_metric(score),
                ", ".join(markers) if markers else "",
                path or "n/a",
            )
        )
    return tuple(rows)


def parse_config_text(config_text: str | None) -> dict[str, Any]:
    if not config_text:
        return {}
    try:
        parsed = yaml.safe_load(config_text)
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def nested_config_value(config: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def format_config_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "none"
    if isinstance(value, dict):
        return ", ".join(f"{key}={format_config_value(val)}" for key, val in value.items())
    return str(value)


def config_summary_rows(
    config: dict[str, Any],
    fields: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (label, format_config_value(nested_config_value(config, path)))
        for label, path in fields
    )


GAME_CONFIG_FIELDS = (
    ("Players", ("game", "players")),
    ("Max Turns", ("game", "max_turns")),
    ("Included Cards", ("game", "include_cards")),
    ("Excluded Cards", ("game", "exclude_cards")),
    ("Combo Rules", ("game", "enabled_combo_rules")),
)
TRAINING_CONFIG_FIELDS = (
    ("Iterations", ("training", "iterations")),
    ("Train Batch Size", ("training", "train_batch_size")),
    ("Minibatch Size", ("training", "minibatch_size")),
    ("Rollout Fragment", ("training", "rollout_fragment_length")),
    ("Epochs", ("training", "num_epochs")),
    ("Learning Rate", ("training", "lr")),
)
EXPERIMENT_CONFIG_FIELDS = (
    ("Checkpoint Interval", ("experiment", "checkpoint_interval")),
    ("Evaluation Interval", ("experiment", "evaluation_interval")),
    ("Evaluation Opponents", ("experiment", "evaluation_opponents")),
)
POLICY_CONFIG_FIELDS = (
    ("Mode", ("policy_setup", "mode")),
    ("Shared Policy", ("policy_setup", "shared_policy_id")),
    ("Seat Policies", ("policy_setup", "seat_policies")),
    ("Trainable Policies", ("policy_setup", "trainable_policies")),
    ("Frozen Policies", ("policy_setup", "frozen_policies")),
    ("Opponent Pool", ("policy_setup", "opponent_pool")),
)
CONFIG_COMPARISON_FIELDS = (
    ("Players", ("game", "players")),
    ("Max Turns", ("game", "max_turns")),
    ("Included Cards", ("game", "include_cards")),
    ("Excluded Cards", ("game", "exclude_cards")),
    ("Combo Rules", ("game", "enabled_combo_rules")),
    ("Iterations", ("training", "iterations")),
    ("Batch Size", ("training", "train_batch_size")),
    ("Learning Rate", ("training", "lr")),
    ("Checkpoint Interval", ("experiment", "checkpoint_interval")),
    ("Evaluation Interval", ("experiment", "evaluation_interval")),
    ("Evaluation Opponents", ("experiment", "evaluation_opponents")),
    ("Policy Mode", ("policy_setup", "mode")),
)


def config_validation_messages(config: dict[str, Any]) -> tuple[str, ...]:
    messages: list[str] = []
    if not config:
        return ("No resolved YAML config was found for this run.",)
    for section in ("game", "training", "experiment", "policy_setup"):
        if not isinstance(config.get(section), dict):
            messages.append(f"Missing `{section}` section.")
    players = optional_int(nested_config_value(config, ("game", "players")))
    if players is not None and players < 2:
        messages.append("Game should have at least two players.")
    include_cards = nested_config_value(config, ("game", "include_cards"))
    exclude_cards = nested_config_value(config, ("game", "exclude_cards"))
    if include_cards and exclude_cards:
        messages.append("Both include_cards and exclude_cards are set.")
    evaluation_interval = optional_int(
        nested_config_value(config, ("experiment", "evaluation_interval"))
    )
    evaluation_episodes = optional_int(
        nested_config_value(config, ("experiment", "evaluation_episodes"))
    )
    if evaluation_interval is not None and evaluation_episodes is not None:
        if evaluation_interval > 0 and evaluation_episodes < 1:
            messages.append("Evaluation interval is enabled but episodes is less than one.")
    return tuple(messages)


def config_diff_rows(
    runs: tuple[DashboardRunSummary, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    configs = [parse_config_text(run.resolved_config_text) for run in runs]
    rows: list[tuple[str, tuple[str, ...]]] = []
    for label, path in CONFIG_COMPARISON_FIELDS:
        values = tuple(
            format_config_value(nested_config_value(config, path))
            for config in configs
        )
        if len(set(values)) > 1:
            rows.append((label, values))
    return tuple(rows)


def run_latest_mean_win_rate(run: DashboardRunSummary) -> float | None:
    if not run.latest_win_rates:
        return None
    return sum(run.latest_win_rates.values()) / len(run.latest_win_rates)


def run_best_evaluation_score(run: DashboardRunSummary) -> float | None:
    best = best_evaluation_checkpoint(run.evaluations, run.checkpoints)
    return best[1] if best is not None else None


def comparison_best_run_rows(
    runs: tuple[DashboardRunSummary, ...],
) -> tuple[tuple[str, str, str, str, str], ...]:
    rows = [
        (
            run.run_name,
            format_metric(run_latest_mean_win_rate(run)),
            format_metric(run_best_evaluation_score(run)),
            format_metric(run.latest_reward_mean),
            format_metric(run.latest_environment_steps_per_second, digits=1),
        )
        for run in runs
    ]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                float("-inf") if row[1] == "n/a" else float(row[1]),
                row[0],
            ),
            reverse=True,
        )
    )


def discover_config_files(config_dir: Path = DEFAULT_CONFIG_DIR) -> tuple[Path, ...]:
    if not config_dir.exists():
        return ()
    return tuple(sorted(config_dir.glob("*.yaml")))


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def read_launcher_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        return {}
    return parse_config_text(config_path.read_text(encoding="utf-8"))


def resolve_launcher_run_name(
    *,
    config_path: Path,
    run_name_override: str | None,
) -> str:
    if run_name_override and run_name_override.strip():
        return run_name_override.strip()
    config = read_launcher_config(config_path)
    dashboard_run_name = nested_config_value(config, ("dashboard_logging", "run_name"))
    experiment_run_name = nested_config_value(config, ("experiment", "run_name"))
    return str(dashboard_run_name or experiment_run_name or config_path.stem)


def launcher_run_dir(
    *,
    experiment_dir: Path,
    run_name: str,
) -> Path:
    return experiment_dir / run_name


def launcher_expected_run_dir(
    *,
    experiment_dir: Path,
    config_path: Path,
    run_name_override: str | None,
) -> Path:
    run_name = resolve_launcher_run_name(
        config_path=config_path,
        run_name_override=run_name_override,
    )
    return launcher_run_dir(experiment_dir=experiment_dir, run_name=run_name)


def launcher_duplicate_run_dir(
    *,
    experiment_dir: Path,
    config_path: Path,
    run_name_override: str | None,
) -> Path | None:
    run_dir = launcher_expected_run_dir(
        experiment_dir=experiment_dir,
        config_path=config_path,
        run_name_override=run_name_override,
    )
    return run_dir if run_dir.exists() else None


def optional_int_argument(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    try:
        return str(int(value))
    except ValueError:
        return None


def build_launcher_command(
    *,
    python_executable: str,
    config_path: Path,
    experiment_dir: Path,
    run_name: str | None = None,
    iterations: str | None = None,
    seed: str | None = None,
    wandb_mode: str = "use-config",
    show_progress: bool = True,
) -> tuple[str, ...]:
    command = [
        python_executable,
        str(REPO_ROOT / "scripts" / "train_rllib_pettingzoo.py"),
        "--config",
        display_path(config_path),
        "--dashboard-logging",
        "--experiment-dir",
        str(experiment_dir),
    ]
    if run_name and run_name.strip():
        cleaned_run_name = run_name.strip()
        command.extend(["--run-name", cleaned_run_name])
        command.extend(["--dashboard-run-name", cleaned_run_name])
    if parsed_iterations := optional_int_argument(iterations):
        command.extend(["--iterations", parsed_iterations])
    if parsed_seed := optional_int_argument(seed):
        command.extend(["--seed", parsed_seed])
    if wandb_mode != "use-config":
        command.extend(["--wandb-mode", wandb_mode])
    if not show_progress:
        command.append("--no-progress")
    return tuple(command)


def format_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def launched_run_selection(
    comparison_run_names: set[str],
    launched_run_name: str,
) -> tuple[str, set[str]]:
    updated_comparison_run_names = set(comparison_run_names)
    updated_comparison_run_names.add(launched_run_name)
    return launched_run_name, updated_comparison_run_names


def refresh_wait_seconds(
    default_refresh_seconds: int,
    *,
    launcher_running: bool,
) -> int:
    if launcher_running:
        return min(default_refresh_seconds, ACTIVE_LAUNCH_REFRESH_SECONDS)
    return default_refresh_seconds


def latest_metric_update(metrics: tuple[dict[str, Any], ...]) -> str:
    if not metrics:
        return "n/a"
    return str(metrics[-1].get("logged_at") or "n/a")


def training_status_rows(
    run: DashboardRunSummary | None,
    launcher_state: dict[str, Any],
) -> tuple[tuple[str, str], ...]:
    rows = [
        ("Launcher Status", str(launcher_state.get("status") or "Idle")),
        ("Expected Directory", str(launcher_state.get("expected_run_dir") or "n/a")),
        ("Metrics Rows", str(len(run.metrics) if run is not None else 0)),
        (
            "Last Metric Update",
            latest_metric_update(run.metrics) if run is not None else "n/a",
        ),
    ]
    if run is not None:
        rows.extend(
            [
                ("Selected Run", run.run_name),
                ("Progress", format_progress(run)),
                ("ETA", format_duration_seconds(run.estimated_seconds_remaining)),
                (
                    "Env Steps/s",
                    format_metric(
                        run.latest_environment_steps_per_second,
                        digits=1,
                    ),
                ),
            ]
        )
    else:
        rows.append(
            (
                "Selected Run",
                str(launcher_state.get("resolved_run_name") or "n/a"),
            )
        )
    return tuple(rows)


def run_dashboard(
    *,
    experiment_dir: Path = DEFAULT_EXPERIMENT_DIR,
    refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
) -> None:
    try:
        import flet as ft
    except ImportError as exc:
        raise RuntimeError(
            'The local dashboard requires `pip install -e ".[ui]"` or '
            "`pip install flet[all] flet-charts`."
        ) from exc

    def app(page: ft.Page) -> None:
        page.title = "Exploding Kittens Experiments"
        page.padding = 18
        page.bgcolor = ft.Colors.GREY_50
        page.theme_mode = ft.ThemeMode.LIGHT
        config_files = discover_config_files()
        default_config_path = display_path(config_files[0]) if config_files else ""
        launcher_lock = threading.Lock()

        state: dict[str, Any] = {
            "runs": (),
            "selected_run_name": None,
            "comparison_run_names": set(),
            "smooth_training": False,
            "launcher_config_path": default_config_path,
            "launcher_run_name": "",
            "launcher_iterations": "",
            "launcher_seed": "",
            "launcher_wandb_mode": "use-config",
            "launcher_show_progress": True,
            "launcher_output_lines": (),
            "launcher_status": "Idle",
            "launcher_process": None,
            "launcher_pending_stop": False,
            "stop_refresh": threading.Event(),
        }

        run_list = ft.ListView(expand=True, spacing=4, padding=0)
        title = ft.Text("No run selected", size=22, weight=ft.FontWeight.BOLD)
        subtitle = ft.Text(
            "Open a training run to populate reports/experiments.",
            size=12,
            color=ft.Colors.GREY_700,
        )
        tabs = ft.Tabs(
            length=7,
            expand=True,
            content=ft.Column(
                expand=True,
                controls=[
                    ft.TabBar(tabs=[]),
                    ft.TabBarView(expand=True, controls=[]),
                ],
            ),
        )

        def selected_run() -> DashboardRunSummary | None:
            selected_name = state["selected_run_name"]
            for run in state["runs"]:
                if run.run_name == selected_name:
                    return run
            return state["runs"][0] if state["runs"] else None

        def select_run(run_name: str) -> None:
            state["selected_run_name"] = run_name
            render()

        def toggle_comparison_run(run_name: str, enabled: bool) -> None:
            comparison_run_names = set(state["comparison_run_names"])
            if enabled:
                comparison_run_names.add(run_name)
            else:
                comparison_run_names.discard(run_name)
            state["comparison_run_names"] = comparison_run_names
            render()

        def set_launcher_field(
            field_name: str,
            value: Any,
            render_now: bool = True,
        ) -> None:
            state[field_name] = value
            if render_now:
                render()

        def selected_launcher_config_path() -> Path:
            path = Path(str(state["launcher_config_path"]))
            return path if path.is_absolute() else REPO_ROOT / path

        def launcher_view_state() -> dict[str, Any]:
            config_path = selected_launcher_config_path()
            run_name_override = str(state["launcher_run_name"])
            command = build_launcher_command(
                python_executable=sys.executable,
                config_path=config_path,
                experiment_dir=experiment_dir,
                run_name=run_name_override,
                iterations=str(state["launcher_iterations"]),
                seed=str(state["launcher_seed"]),
                wandb_mode=str(state["launcher_wandb_mode"]),
                show_progress=bool(state["launcher_show_progress"]),
            )
            duplicate_run_dir = launcher_duplicate_run_dir(
                experiment_dir=experiment_dir,
                config_path=config_path,
                run_name_override=run_name_override,
            )
            expected_run_dir = launcher_expected_run_dir(
                experiment_dir=experiment_dir,
                config_path=config_path,
                run_name_override=run_name_override,
            )
            process = state["launcher_process"]
            is_running = process is not None and process.poll() is None
            return {
                "config_files": config_files,
                "config_path": str(state["launcher_config_path"]),
                "run_name": str(state["launcher_run_name"]),
                "resolved_run_name": resolve_launcher_run_name(
                    config_path=config_path,
                    run_name_override=run_name_override,
                ),
                "expected_run_dir": expected_run_dir,
                "iterations": str(state["launcher_iterations"]),
                "seed": str(state["launcher_seed"]),
                "wandb_mode": str(state["launcher_wandb_mode"]),
                "show_progress": bool(state["launcher_show_progress"]),
                "command": command,
                "command_preview": format_command(command),
                "duplicate_run_dir": duplicate_run_dir,
                "output_lines": tuple(state["launcher_output_lines"]),
                "status": str(state["launcher_status"]),
                "is_running": is_running,
                "pending_stop": bool(state["launcher_pending_stop"]),
            }

        def append_launcher_output(line: str) -> None:
            with launcher_lock:
                lines = list(state["launcher_output_lines"])
                lines.append(line)
                state["launcher_output_lines"] = tuple(lines[-MAX_LAUNCHER_LOG_LINES:])

        def start_launcher(_: Any | None = None) -> None:
            view_state = launcher_view_state()
            config_path = selected_launcher_config_path()
            if not config_path.is_file():
                append_launcher_output(f"Config file not found: {config_path}")
                render()
                return
            if view_state["duplicate_run_dir"] is not None:
                append_launcher_output(
                    f"Run directory already exists: {view_state['duplicate_run_dir']}"
                )
                render()
                return
            if view_state["is_running"]:
                return
            command = tuple(view_state["command"])
            launched_run_name = str(view_state["resolved_run_name"])
            (
                state["selected_run_name"],
                state["comparison_run_names"],
            ) = launched_run_selection(
                set(state["comparison_run_names"]),
                launched_run_name,
            )
            append_launcher_output(f"> {format_command(command)}")
            try:
                process = subprocess.Popen(
                    list(command),
                    cwd=REPO_ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
            except OSError as exc:
                state["launcher_status"] = "Failed to start"
                append_launcher_output(f"Launch failed: {exc}")
                render()
                return
            state["launcher_process"] = process
            state["launcher_status"] = "Running"
            state["launcher_pending_stop"] = False
            threading.Thread(
                target=stream_launcher_output,
                args=(process,),
                daemon=True,
            ).start()
            render()

        def stream_launcher_output(process: subprocess.Popen[str]) -> None:
            if process.stdout is not None:
                for line in process.stdout:
                    append_launcher_output(line.rstrip())
                    render()
            return_code = process.wait()
            with launcher_lock:
                if state["launcher_process"] is process:
                    state["launcher_process"] = None
                state["launcher_pending_stop"] = False
                state["launcher_status"] = (
                    "Completed" if return_code == 0 else f"Exited {return_code}"
                )
            append_launcher_output(f"Process finished with exit code {return_code}.")
            refresh_runs()

        def request_stop_launcher(_: Any | None = None) -> None:
            state["launcher_pending_stop"] = True
            render()

        def cancel_stop_launcher(_: Any | None = None) -> None:
            state["launcher_pending_stop"] = False
            render()

        def confirm_stop_launcher(_: Any | None = None) -> None:
            process = state["launcher_process"]
            if process is not None and process.poll() is None:
                state["launcher_status"] = "Stopping"
                append_launcher_output("Stop requested.")
                process.terminate()
            state["launcher_pending_stop"] = False
            render()

        def refresh_runs(_: Any | None = None) -> None:
            runs = load_dashboard_runs(experiment_dir)
            state["runs"] = runs
            selected_name = state["selected_run_name"]
            valid_run_names = {run.run_name for run in runs}
            launcher_view = launcher_view_state()
            launched_run_pending = (
                bool(launcher_view["is_running"])
                and selected_name == launcher_view["resolved_run_name"]
            )
            if selected_name not in valid_run_names and not launched_run_pending:
                state["selected_run_name"] = runs[0].run_name if runs else None
            comparison_run_names = set(state["comparison_run_names"]) & valid_run_names
            if launched_run_pending and selected_name is not None:
                comparison_run_names.add(str(selected_name))
            if not comparison_run_names and runs:
                comparison_run_names = {run.run_name for run in runs[:2]}
            state["comparison_run_names"] = comparison_run_names
            render()

        def render() -> None:
            run_list.controls.clear()
            if state["runs"]:
                for run in state["runs"]:
                    comparison_run_names = set(state["comparison_run_names"])
                    run_list.controls.append(
                        ft.Row(
                            controls=[
                                ft.Checkbox(
                                    value=run.run_name in comparison_run_names,
                                    on_change=(
                                        lambda event, name=run.run_name: (
                                            toggle_comparison_run(
                                                name,
                                                bool(event.control.value),
                                            )
                                        )
                                    ),
                                ),
                                ft.TextButton(
                                    content=f"{run.run_name} [{run.status}]",
                                    tooltip=str(run.run_dir),
                                    on_click=(
                                        lambda _, name=run.run_name: select_run(name)
                                    ),
                                ),
                            ],
                            spacing=4,
                        )
                    )
            else:
                run_list.controls.append(
                    ft.Text(
                        "No local experiment runs found.",
                        color=ft.Colors.GREY_700,
                    )
                )

            run = selected_run()
            pending_selected_name = state["selected_run_name"]
            title.value = (
                run.run_name
                if run is not None
                else str(pending_selected_name or "No run selected")
            )
            subtitle.value = (
                selected_run_subtitle(run)
                if run is not None
                else (
                    f"Waiting for logs in {experiment_dir}"
                    if pending_selected_name
                    else f"Watching {experiment_dir}"
                )
            )
            labels, contents = build_tabs(
                ft,
                run,
                runs=state["runs"],
                comparison_run_names=set(state["comparison_run_names"]),
                launcher_state=launcher_view_state(),
                launcher_callbacks={
                    "set_field": set_launcher_field,
                    "start": start_launcher,
                    "request_stop": request_stop_launcher,
                    "confirm_stop": confirm_stop_launcher,
                    "cancel_stop": cancel_stop_launcher,
                },
                smooth_training=state["smooth_training"],
                on_smooth_training_change=toggle_smooth_training,
            )
            tabs.content.controls[0].tabs = labels
            tabs.content.controls[1].controls = contents
            page.update()

        def toggle_smooth_training(event: Any) -> None:
            state["smooth_training"] = bool(event.control.value)
            render()

        def auto_refresh_loop() -> None:
            while True:
                process = state["launcher_process"]
                launcher_running = process is not None and process.poll() is None
                wait_seconds = refresh_wait_seconds(
                    refresh_seconds,
                    launcher_running=launcher_running,
                )
                if state["stop_refresh"].wait(wait_seconds):
                    break
                refresh_runs()

        def close_dashboard(_: Any) -> None:
            state["stop_refresh"].set()
            process = state["launcher_process"]
            if process is not None and process.poll() is None:
                process.terminate()

        page.on_close = close_dashboard
        refresh_button = ft.IconButton(
            icon=ft.Icons.REFRESH,
            tooltip="Refresh runs",
            on_click=refresh_runs,
        )
        left_panel = ft.Container(
            width=300,
            padding=14,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Column(
                controls=[
                    ft.Row(
                        controls=[
                            ft.Text(
                                "Runs",
                                size=16,
                                weight=ft.FontWeight.BOLD,
                                expand=True,
                            ),
                            refresh_button,
                        ]
                    ),
                    ft.Text(str(experiment_dir), size=12, color=ft.Colors.GREY_700),
                    run_list,
                ],
                expand=True,
            ),
        )
        main_panel = ft.Column(
            controls=[
                ft.Container(
                    padding=14,
                    border=border_all(ft),
                    border_radius=8,
                    bgcolor=ft.Colors.WHITE,
                    content=ft.Column(controls=[title, subtitle], spacing=2),
                ),
                tabs,
            ],
            expand=True,
        )
        page.add(ft.Row(controls=[left_panel, main_panel], expand=True, spacing=18))
        refresh_runs()
        threading.Thread(target=auto_refresh_loop, daemon=True).start()

    ft.run(app)


def selected_run_subtitle(run: DashboardRunSummary) -> str:
    stale = " stale" if run.is_stale else ""
    return (
        f"status={run.status}{stale} progress={format_progress(run)} "
        f"reward={format_metric(run.latest_reward_mean)} "
        f"env_steps/s={format_metric(run.latest_environment_steps_per_second, digits=1)}"
    )


def build_tabs(
    ft: Any,
    run: DashboardRunSummary | None,
    *,
    runs: tuple[DashboardRunSummary, ...] = (),
    comparison_run_names: set[str] | None = None,
    launcher_state: dict[str, Any] | None = None,
    launcher_callbacks: dict[str, Any] | None = None,
    smooth_training: bool = False,
    on_smooth_training_change: Any | None = None,
) -> tuple[list[Any], list[Any]]:
    labels = [
        ft.Tab(label="Overview"),
        ft.Tab(label="Training"),
        ft.Tab(label="Evaluation"),
        ft.Tab(label="Behavior"),
        ft.Tab(label="Checkpoints"),
        ft.Tab(label="Config"),
        ft.Tab(label="Comparison"),
    ]
    comparison_run_names = comparison_run_names or set()
    comparison_runs = tuple(
        selected_run
        for selected_run in runs
        if selected_run.run_name in comparison_run_names
    )
    contents = [
        overview_tab(ft, run),
        training_tab(
            ft,
            run,
            launcher_state=launcher_state or {},
            launcher_callbacks=launcher_callbacks or {},
            smooth=smooth_training,
            on_smooth_change=on_smooth_training_change,
        ),
        evaluation_tab(ft, run, smooth=smooth_training),
        behavior_tab(ft, run, smooth=smooth_training),
        checkpoints_tab(ft, run),
        config_tab(ft, run),
        comparison_tab(ft, comparison_runs, smooth=smooth_training),
    ]
    return labels, contents


def overview_tab(ft: Any, run: DashboardRunSummary | None) -> Any:
    if run is None:
        return empty_tab(ft)
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Row(
                    controls=[
                        overview_metric_tile(
                            ft,
                            "Status",
                            stale_status_text(run),
                        ),
                        overview_metric_tile(ft, "Progress", format_progress(run)),
                        overview_metric_tile(
                            ft,
                            "Elapsed",
                            format_duration_seconds(run.elapsed_seconds),
                        ),
                        overview_metric_tile(
                            ft,
                            "ETA",
                            format_duration_seconds(run.estimated_seconds_remaining),
                        ),
                    ],
                    spacing=10,
                ),
                ft.Row(
                    controls=[
                        overview_metric_tile(
                            ft,
                            "Reward Mean",
                            format_metric(run.latest_reward_mean),
                        ),
                        overview_metric_tile(
                            ft,
                            "Episode Length",
                            format_metric(run.latest_episode_length_mean),
                        ),
                        overview_metric_tile(
                            ft,
                            "Env Steps/s",
                            format_metric(
                                run.latest_environment_steps_per_second,
                                digits=1,
                            ),
                        ),
                    ],
                    spacing=10,
                ),
                ft.Text("Latest Win Rates", size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    controls=[
                        overview_metric_tile(
                            ft,
                            opponent,
                            format_metric(run.latest_win_rates.get(opponent)),
                        )
                        for opponent in ("random", "safe-rule", "draw-only")
                    ],
                    spacing=10,
                ),
                ft.Text("Run Details", size=16, weight=ft.FontWeight.BOLD),
                detail_row(ft, "Run", run.run_name),
                detail_row(ft, "Config", run.active_config_name or "n/a"),
                detail_row(ft, "Last Updated", run.last_updated_at or "n/a"),
                detail_row(ft, "Latest Checkpoint", run.latest_checkpoint_path or "n/a"),
                detail_row(ft, "Run Directory", str(run.run_dir)),
            ],
            spacing=12,
        ),
    )


def training_tab(
    ft: Any,
    run: DashboardRunSummary | None,
    *,
    launcher_state: dict[str, Any],
    launcher_callbacks: dict[str, Any],
    smooth: bool,
    on_smooth_change: Any | None,
) -> Any:
    controls: list[Any] = [
        ft.Row(
            controls=[
                ft.Text("Training", size=16, weight=ft.FontWeight.BOLD, expand=True),
                ft.Switch(
                    label="Smooth",
                    value=smooth,
                    on_change=on_smooth_change,
                ),
            ],
            spacing=16,
        ),
        ft.Row(
            controls=[
                launcher_control_panel(ft, launcher_state, launcher_callbacks),
                training_status_panel(ft, run, launcher_state),
            ],
            spacing=12,
        ),
    ]
    if run is None or (not run.metrics and not run.evaluations):
        controls.append(
            ft.Container(
                padding=12,
                border=border_all(ft),
                border_radius=8,
                bgcolor=ft.Colors.WHITE,
                content=ft.Text(
                    "Training metrics will appear here as soon as the run writes "
                    "local dashboard logs."
                ),
            )
        )
    else:
        controls.extend(
            [
                ft.Text(
                    "Latest Baseline Win Rates",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                ),
                ft.Row(
                    controls=[
                        overview_metric_tile(
                            ft,
                            opponent,
                            format_metric(run.latest_win_rates.get(opponent)),
                        )
                        for opponent in ("random", "safe-rule", "draw-only")
                    ],
                    spacing=10,
                ),
                ft.Row(
                    controls=[
                        win_rate_chart(
                            ft,
                            evaluations=run.evaluations,
                            smooth=smooth,
                        ),
                        evaluation_metric_chart(
                            ft,
                            title="Illegal Action Rate",
                            evaluations=run.evaluations,
                            metric_name="illegal_action_rate",
                            smooth=smooth,
                        ),
                    ],
                    spacing=12,
                ),
                ft.Row(
                    controls=[
                        training_chart(
                            ft,
                            title="Environment Steps",
                            metrics=run.metrics,
                            metric_name="environment_steps_sampled",
                            smooth=smooth,
                            color=ft.Colors.ORANGE,
                        ),
                        training_chart(
                            ft,
                            title="Environment Steps/s",
                            metrics=run.metrics,
                            metric_name="environment_steps_per_second",
                            smooth=smooth,
                            color=ft.Colors.PURPLE,
                        ),
                    ],
                    spacing=12,
                ),
                ft.Text(
                    "Evaluation Summary",
                    size=16,
                    weight=ft.FontWeight.BOLD,
                ),
                latest_evaluation_metrics_table(ft, run.evaluations),
                ft.Text("Latest Metrics", size=16, weight=ft.FontWeight.BOLD),
                latest_training_metrics_table(ft, run.metrics),
            ]
        )
    return ft.Container(
        padding=12,
        content=ft.Column(
            controls=controls,
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def evaluation_tab(
    ft: Any,
    run: DashboardRunSummary | None,
    *,
    smooth: bool,
) -> Any:
    if run is None:
        return empty_tab(ft)
    if not run.evaluations:
        return ft.Container(
            padding=16,
            content=ft.Column(
                controls=[
                    ft.Text("Evaluation", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("No evaluation rows have been logged yet."),
                ],
                spacing=8,
            ),
        )
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text("Evaluation", size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    controls=[
                        overview_metric_tile(
                            ft,
                            opponent,
                            format_metric(run.latest_win_rates.get(opponent)),
                        )
                        for opponent in ("random", "safe-rule", "draw-only")
                    ],
                    spacing=10,
                ),
                ft.Row(
                    controls=[
                        win_rate_chart(
                            ft,
                            evaluations=run.evaluations,
                            smooth=smooth,
                        ),
                        scripted_opponent_win_rate_chart(
                            ft,
                            evaluations=run.evaluations,
                            smooth=smooth,
                        ),
                    ],
                    spacing=12,
                ),
                ft.Row(
                    controls=[
                        evaluation_metric_chart(
                            ft,
                            title="Illegal Action Rate",
                            evaluations=run.evaluations,
                            metric_name="illegal_action_rate",
                            smooth=smooth,
                            y_axis_label="Illegal action rate",
                        ),
                        evaluation_metric_chart(
                            ft,
                            title="Seat-Order Advantage",
                            evaluations=run.evaluations,
                            metric_name="seat_order_advantage",
                            smooth=smooth,
                            y_axis_label="Win-rate spread",
                        ),
                    ],
                    spacing=12,
                ),
                ft.Text("Best Checkpoint", size=16, weight=ft.FontWeight.BOLD),
                best_checkpoint_card(ft, run),
                ft.Text("Latest Evaluation", size=16, weight=ft.FontWeight.BOLD),
                latest_evaluation_metrics_table(ft, run.evaluations),
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def behavior_tab(
    ft: Any,
    run: DashboardRunSummary | None,
    *,
    smooth: bool,
) -> Any:
    if run is None:
        return empty_tab(ft)
    if not run.evaluations:
        return ft.Container(
            padding=16,
            content=ft.Column(
                controls=[
                    ft.Text("Behavior", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("No behavior diagnostics have been logged yet."),
                ],
                spacing=8,
            ),
        )
    action_counts = action_category_counts(run.evaluations)
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text("Behavior", size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    controls=[
                        overview_metric_tile(ft, label.title(), str(count))
                        for label, count in action_counts.items()
                    ],
                    spacing=10,
                ),
                ft.Row(
                    controls=[
                        combo_count_chart(
                            ft,
                            evaluations=run.evaluations,
                            smooth=smooth,
                        ),
                        evaluation_metric_chart(
                            ft,
                            title="Illegal Action Rate",
                            evaluations=run.evaluations,
                            metric_name="illegal_action_rate",
                            smooth=smooth,
                            y_axis_label="Illegal action rate",
                        ),
                    ],
                    spacing=12,
                ),
                ft.Text("Combo Rate", size=16, weight=ft.FontWeight.BOLD),
                combo_rate_table(ft, run.evaluations),
                ft.Text("Action Distribution", size=16, weight=ft.FontWeight.BOLD),
                latest_action_distribution_table(ft, run.evaluations),
                ft.Text("Cards Played", size=16, weight=ft.FontWeight.BOLD),
                latest_card_distribution_table(ft, run.evaluations),
                ft.Text("Suspicious Behavior", size=16, weight=ft.FontWeight.BOLD),
                suspicious_behavior_panel(ft, run.evaluations),
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def checkpoints_tab(ft: Any, run: DashboardRunSummary | None) -> Any:
    if run is None:
        return empty_tab(ft)
    if not run.checkpoints:
        return ft.Container(
            padding=16,
            content=ft.Column(
                controls=[
                    ft.Text("Checkpoints", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("No checkpoints have been logged yet."),
                ],
                spacing=8,
            ),
        )
    config = parse_config_text(run.resolved_config_text)
    players = optional_int(nested_config_value(config, ("game", "players"))) or 3
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text("Checkpoints", size=16, weight=ft.FontWeight.BOLD),
                best_checkpoint_card(ft, run),
                checkpoints_table(ft, run, players=players),
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def config_tab(ft: Any, run: DashboardRunSummary | None) -> Any:
    if run is None:
        return empty_tab(ft)
    config = parse_config_text(run.resolved_config_text)
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text("Config", size=16, weight=ft.FontWeight.BOLD),
                ft.Row(
                    controls=[
                        config_summary_table(
                            ft,
                            "Game",
                            config_summary_rows(config, GAME_CONFIG_FIELDS),
                        ),
                        config_summary_table(
                            ft,
                            "Training",
                            config_summary_rows(config, TRAINING_CONFIG_FIELDS),
                        ),
                    ],
                    spacing=12,
                ),
                ft.Row(
                    controls=[
                        config_summary_table(
                            ft,
                            "Experiment",
                            config_summary_rows(config, EXPERIMENT_CONFIG_FIELDS),
                        ),
                        config_summary_table(
                            ft,
                            "Policy",
                            config_summary_rows(config, POLICY_CONFIG_FIELDS),
                        ),
                    ],
                    spacing=12,
                ),
                ft.Text("Validation", size=16, weight=ft.FontWeight.BOLD),
                config_validation_panel(ft, config),
                ft.Text("Resolved YAML", size=16, weight=ft.FontWeight.BOLD),
                ft.Container(
                    padding=12,
                    border=border_all(ft),
                    border_radius=8,
                    content=ft.Text(
                        run.resolved_config_text or "No resolved config found.",
                        font_family="Consolas",
                        size=12,
                        selectable=True,
                    ),
                ),
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def comparison_tab(
    ft: Any,
    runs: tuple[DashboardRunSummary, ...],
    *,
    smooth: bool,
) -> Any:
    if len(runs) < 2:
        return ft.Container(
            padding=16,
            content=ft.Column(
                controls=[
                    ft.Text("Comparison", size=16, weight=ft.FontWeight.BOLD),
                    ft.Text("Select at least two runs in the left panel."),
                ],
                spacing=8,
            ),
        )
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text("Comparison", size=16, weight=ft.FontWeight.BOLD),
                comparison_best_runs_table(ft, runs),
                ft.Row(
                    controls=[
                        comparison_training_chart(
                            ft,
                            title="Reward Mean",
                            runs=runs,
                            metric_name="episode_reward_mean",
                            y_axis_label="Reward mean",
                            smooth=smooth,
                        ),
                        comparison_training_chart(
                            ft,
                            title="Environment Steps/s",
                            runs=runs,
                            metric_name="environment_steps_per_second",
                            y_axis_label="Env steps/s",
                            smooth=smooth,
                        ),
                    ],
                    spacing=12,
                ),
                ft.Row(
                    controls=[
                        comparison_win_rate_chart(
                            ft,
                            runs=runs,
                            opponent="random",
                            smooth=smooth,
                        ),
                        comparison_win_rate_chart(
                            ft,
                            runs=runs,
                            opponent="safe-rule",
                            smooth=smooth,
                        ),
                    ],
                    spacing=12,
                ),
                comparison_win_rate_chart(
                    ft,
                    runs=runs,
                    opponent="draw-only",
                    smooth=smooth,
                ),
                ft.Text("Config Differences", size=16, weight=ft.FontWeight.BOLD),
                config_diff_table(ft, runs),
            ],
            spacing=12,
            scroll=ft.ScrollMode.AUTO,
        ),
    )


def compact_text_field(
    ft: Any,
    *,
    value: str,
    label: str | None = None,
    on_change: Any | None = None,
    on_blur: Any | None = None,
    on_submit: Any | None = None,
    read_only: bool = False,
    multiline: bool = False,
    min_lines: int | None = None,
    max_lines: int | None = None,
    width: int | None = None,
    expand: bool | None = None,
) -> Any:
    return ft.TextField(
        label=label,
        value=value,
        on_change=on_change,
        on_blur=on_blur,
        on_submit=on_submit,
        read_only=read_only,
        multiline=multiline,
        min_lines=min_lines,
        max_lines=max_lines,
        width=width,
        expand=expand,
        dense=True,
        text_size=12,
        content_padding=ft.Padding(left=9, top=5, right=9, bottom=5),
    )


def compact_dropdown(
    ft: Any,
    *,
    label: str,
    value: str,
    options: list[Any],
    on_select: Any | None,
    width: int | None = None,
    expand: bool | None = None,
) -> Any:
    return ft.Dropdown(
        label=label,
        value=value,
        options=options,
        on_select=on_select,
        width=width,
        expand=expand,
        dense=True,
        text_size=12,
        content_padding=ft.Padding(left=9, top=5, right=9, bottom=5),
    )


def compact_kv_text(
    ft: Any,
    label: str,
    value: str,
    *,
    expand: bool | None = None,
) -> Any:
    return ft.Container(
        expand=expand,
        padding=6,
        border_radius=8,
        bgcolor=ft.Colors.GREY_50,
        content=ft.Row(
            controls=[
                ft.Text(label, size=11, color=ft.Colors.GREY_600),
                ft.Text(value, size=11, selectable=True, expand=True),
            ],
            spacing=6,
        ),
    )


def launcher_control_panel(
    ft: Any,
    launcher_state: dict[str, Any],
    callbacks: dict[str, Any],
) -> Any:
    set_field = callbacks.get("set_field", lambda *_: None)
    config_files = tuple(launcher_state.get("config_files") or ())
    config_path = str(launcher_state.get("config_path") or "")
    duplicate_run_dir = launcher_state.get("duplicate_run_dir")
    is_running = bool(launcher_state.get("is_running"))
    start_disabled = is_running or not config_path or duplicate_run_dir is not None
    output = "\n".join(launcher_state.get("output_lines") or ())
    config_options = [
        ft.DropdownOption(key=display_path(path), text=path.name)
        for path in config_files
    ]
    controls: list[Any] = [
        ft.Row(
            controls=[
                ft.Text(
                    "Experiment Control",
                    size=14,
                    weight=ft.FontWeight.BOLD,
                    expand=True,
                ),
                launcher_status_chip(ft, str(launcher_state.get("status") or "Idle")),
                ft.Button(
                    content="Start",
                    icon=ft.Icons.PLAY_ARROW,
                    on_click=callbacks.get("start"),
                    disabled=start_disabled,
                ),
                ft.OutlinedButton(
                    content="Stop",
                    icon=ft.Icons.STOP,
                    on_click=callbacks.get("request_stop"),
                    disabled=not is_running,
                ),
                ft.IconButton(
                    icon=ft.Icons.CONTENT_COPY,
                    tooltip="Copy command",
                    on_click=copy_text_handler(
                        str(launcher_state.get("command_preview") or "")
                    ),
                ),
            ],
            spacing=8,
        ),
        ft.Row(
            controls=[
                compact_dropdown(
                    ft,
                    label="Config",
                    value=config_path,
                    options=config_options,
                    on_select=lambda event: set_field(
                        "launcher_config_path",
                        event.control.value,
                    ),
                    expand=True,
                ),
                compact_text_field(
                    ft,
                    label="Run name",
                    value=str(launcher_state.get("run_name") or ""),
                    on_change=lambda event: set_field(
                        "launcher_run_name",
                        event.control.value,
                        False,
                    ),
                    on_blur=lambda event: set_field(
                        "launcher_run_name",
                        event.control.value,
                    ),
                    on_submit=lambda event: set_field(
                        "launcher_run_name",
                        event.control.value,
                    ),
                    width=220,
                ),
                compact_text_field(
                    ft,
                    label="Iterations",
                    value=str(launcher_state.get("iterations") or ""),
                    on_change=lambda event: set_field(
                        "launcher_iterations",
                        event.control.value,
                        False,
                    ),
                    on_blur=lambda event: set_field(
                        "launcher_iterations",
                        event.control.value,
                    ),
                    on_submit=lambda event: set_field(
                        "launcher_iterations",
                        event.control.value,
                    ),
                    width=105,
                ),
                compact_text_field(
                    ft,
                    label="Seed",
                    value=str(launcher_state.get("seed") or ""),
                    on_change=lambda event: set_field(
                        "launcher_seed",
                        event.control.value,
                        False,
                    ),
                    on_blur=lambda event: set_field(
                        "launcher_seed",
                        event.control.value,
                    ),
                    on_submit=lambda event: set_field(
                        "launcher_seed",
                        event.control.value,
                    ),
                    width=95,
                ),
                compact_dropdown(
                    ft,
                    label="WandB",
                    value=str(launcher_state.get("wandb_mode") or "use-config"),
                    options=[
                        ft.DropdownOption(key=mode, text=mode)
                        for mode in WANDB_MODE_OPTIONS
                    ],
                    on_select=lambda event: set_field(
                        "launcher_wandb_mode",
                        event.control.value,
                    ),
                    width=135,
                ),
                ft.Switch(
                    label="Progress",
                    value=bool(launcher_state.get("show_progress", True)),
                    on_change=lambda event: set_field(
                        "launcher_show_progress",
                        bool(event.control.value),
                    ),
                ),
            ],
            spacing=8,
        ),
        ft.Row(
            controls=[
                compact_kv_text(
                    ft,
                    "Run",
                    str(launcher_state.get("resolved_run_name") or "n/a"),
                ),
                compact_kv_text(
                    ft,
                    "Dir",
                    str(launcher_state.get("expected_run_dir") or "n/a"),
                    expand=True,
                ),
            ],
            spacing=8,
        ),
    ]
    if duplicate_run_dir is not None:
        controls.append(
            ft.Container(
                padding=12,
                border=border_all(ft, color=ft.Colors.AMBER_300),
                border_radius=8,
                bgcolor=ft.Colors.AMBER_50,
                content=ft.Text(
                    f"Run directory already exists: {duplicate_run_dir}",
                    color=ft.Colors.AMBER_900,
                ),
            )
        )
    if bool(launcher_state.get("pending_stop")):
        controls.append(
            ft.Container(
                padding=12,
                border=border_all(ft, color=ft.Colors.RED_200),
                border_radius=8,
                bgcolor=ft.Colors.RED_50,
                content=ft.Row(
                    controls=[
                        ft.Text("Stop the active training process?", expand=True),
                        ft.OutlinedButton(
                            content="Cancel",
                            on_click=callbacks.get("cancel_stop"),
                        ),
                        ft.Button(
                            content="Stop",
                            icon=ft.Icons.STOP,
                            on_click=callbacks.get("confirm_stop"),
                        ),
                    ],
                    spacing=10,
                ),
            )
        )
    controls.extend(
        [
            compact_text_field(
                ft,
                value=str(launcher_state.get("command_preview") or ""),
                read_only=True,
                multiline=True,
                min_lines=2,
                max_lines=2,
            ),
            compact_text_field(
                ft,
                value=output or "No launcher output yet.",
                read_only=True,
                multiline=True,
                min_lines=5,
                max_lines=5,
            ),
        ]
    )
    return ft.Container(
        expand=2,
        padding=10,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=controls,
            spacing=8,
        ),
    )


def training_status_panel(
    ft: Any,
    run: DashboardRunSummary | None,
    launcher_state: dict[str, Any],
) -> Any:
    rows = training_status_rows(run, launcher_state)
    return ft.Container(
        expand=1,
        padding=10,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=[
                ft.Text("Live Run Status", size=14, weight=ft.FontWeight.BOLD),
                *[
                    compact_kv_text(ft, metric, value)
                    for metric, value in rows
                ],
            ],
            spacing=6,
        ),
    )


def win_rate_chart(
    ft: Any,
    *,
    evaluations: tuple[dict[str, Any], ...],
    smooth: bool,
) -> Any:
    opponent_series = (
        ("random", ft.Colors.BLUE),
        ("safe-rule", ft.Colors.GREEN),
        ("draw-only", ft.Colors.ORANGE),
    )
    series = []
    for opponent, color in opponent_series:
        points = evaluation_points(
            evaluations,
            opponent,
            "rllib_combined_win_rate",
        )
        if smooth:
            points = smooth_points(points)
        if points:
            series.append((opponent, points, color))
    return line_chart_card(
        ft,
        title="RLlib Win Rate vs Baselines",
        subtitle="Higher is better. Lines are grouped by evaluation opponent.",
        series=tuple(series),
        empty_text="No evaluation win-rate data yet.",
        x_axis_label="Evaluation iteration",
        y_axis_label="RLlib win rate",
        y_min=0.0,
        y_max=1.0,
    )


def scripted_opponent_win_rate_chart(
    ft: Any,
    *,
    evaluations: tuple[dict[str, Any], ...],
    smooth: bool,
) -> Any:
    return evaluation_metric_chart(
        ft,
        title="Scripted Opponent Win Rate",
        evaluations=evaluations,
        metric_name="scripted_opponent_win_rate",
        smooth=smooth,
        y_axis_label="Opponent win rate",
    )


def evaluation_metric_chart(
    ft: Any,
    *,
    title: str,
    evaluations: tuple[dict[str, Any], ...],
    metric_name: str,
    smooth: bool,
    y_axis_label: str = "Rate",
) -> Any:
    opponent_series = (
        ("random", ft.Colors.BLUE),
        ("safe-rule", ft.Colors.GREEN),
        ("draw-only", ft.Colors.ORANGE),
    )
    series = []
    for opponent, opponent_color in opponent_series:
        points = evaluation_points(evaluations, opponent, metric_name)
        if smooth:
            points = smooth_points(points)
        if points:
            series.append((opponent, points, opponent_color))
    return line_chart_card(
        ft,
        title=title,
        subtitle="Evaluation diagnostics grouped by baseline opponent.",
        series=tuple(series),
        empty_text="No evaluation diagnostic data yet.",
        x_axis_label="Evaluation iteration",
        y_axis_label=y_axis_label,
        y_min=0.0,
        y_max=1.0,
    )


def combo_count_chart(
    ft: Any,
    *,
    evaluations: tuple[dict[str, Any], ...],
    smooth: bool,
) -> Any:
    opponent_series = (
        ("random", ft.Colors.BLUE),
        ("safe-rule", ft.Colors.GREEN),
        ("draw-only", ft.Colors.ORANGE),
    )
    series = []
    for opponent, opponent_color in opponent_series:
        points = evaluation_points(evaluations, opponent, "combo_count")
        if smooth:
            points = smooth_points(points)
        if points:
            series.append((opponent, points, opponent_color))
    return line_chart_card(
        ft,
        title="Combo Count",
        subtitle="Combo actions recorded during checkpoint evaluation.",
        series=tuple(series),
        empty_text="No combo count data yet.",
        x_axis_label="Evaluation iteration",
        y_axis_label="Combos",
    )


def training_chart(
    ft: Any,
    *,
    title: str,
    metrics: tuple[dict[str, Any], ...],
    metric_name: str,
    smooth: bool,
    color: Any,
) -> Any:
    points = metric_points(metrics, metric_name)
    if smooth:
        points = smooth_points(points)
    return line_chart_card(
        ft,
        title=title,
        subtitle="Training metrics logged after each RLlib iteration.",
        series=((title, points, color),) if points else (),
        empty_text="No training data yet.",
        x_axis_label="Training iteration",
        y_axis_label=title,
    )


def comparison_training_chart(
    ft: Any,
    *,
    title: str,
    runs: tuple[DashboardRunSummary, ...],
    metric_name: str,
    y_axis_label: str,
    smooth: bool,
) -> Any:
    series = []
    for index, run in enumerate(runs):
        points = metric_points(run.metrics, metric_name)
        if smooth:
            points = smooth_points(points)
        if points:
            series.append((run.run_name, points, chart_color(ft, index)))
    return line_chart_card(
        ft,
        title=title,
        subtitle="Selected runs aligned by training iteration.",
        series=tuple(series),
        empty_text="No comparable training data yet.",
        x_axis_label="Training iteration",
        y_axis_label=y_axis_label,
    )


def comparison_win_rate_chart(
    ft: Any,
    *,
    runs: tuple[DashboardRunSummary, ...],
    opponent: str,
    smooth: bool,
) -> Any:
    series = []
    for index, run in enumerate(runs):
        points = evaluation_points(
            run.evaluations,
            opponent,
            "rllib_combined_win_rate",
        )
        if smooth:
            points = smooth_points(points)
        if points:
            series.append((run.run_name, points, chart_color(ft, index)))
    return line_chart_card(
        ft,
        title=f"Win Rate vs {opponent}",
        subtitle="Selected runs aligned by evaluation iteration.",
        series=tuple(series),
        empty_text=f"No {opponent} evaluation data yet.",
        x_axis_label="Evaluation iteration",
        y_axis_label="RLlib win rate",
        y_min=0.0,
        y_max=1.0,
    )


def chart_color(ft: Any, index: int) -> Any:
    colors = (
        ft.Colors.BLUE,
        ft.Colors.GREEN,
        ft.Colors.ORANGE,
        ft.Colors.PURPLE,
        ft.Colors.RED,
        ft.Colors.TEAL,
    )
    return colors[index % len(colors)]


def line_chart_card(
    ft: Any,
    *,
    title: str,
    subtitle: str,
    series: tuple[tuple[str, tuple[tuple[float, float], ...], Any], ...],
    empty_text: str,
    x_axis_label: str,
    y_axis_label: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> Any:
    try:
        import flet_charts as fc
    except ImportError:
        return ft.Container(
            expand=True,
            padding=10,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Text("Install flet-charts to show charts."),
        )
    if not series:
        return ft.Container(
            expand=True,
            height=220,
            padding=10,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Column(
                controls=[
                    ft.Text(title, weight=ft.FontWeight.BOLD),
                    ft.Text(empty_text),
                ]
            ),
        )
    all_points = tuple(point for _, points, _ in series for point in points)
    min_x, max_x, min_y, max_y = chart_bounds(all_points)
    if y_min is not None:
        min_y = y_min
    if y_max is not None:
        max_y = y_max
    data_series = []
    for label, points, color in series:
        chart_points = [
            fc.LineChartDataPoint(
                x=x_value,
                y=y_value,
                tooltip=f"{label} {x_value:g}: {y_value:g}",
            )
            for x_value, y_value in points
        ]
        data_series.append(
            fc.LineChartData(
                points=chart_points,
                color=color,
                stroke_width=2,
                point=True,
            )
        )
    chart = fc.LineChart(
        height=200,
        data_series=data_series,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        left_axis=fc.ChartAxis(
            title=ft.Text(y_axis_label, size=11, color=ft.Colors.GREY_700),
            title_size=34,
            show_labels=True,
            label_size=34,
            labels=[
                fc.ChartAxisLabel(
                    value=value,
                    label=ft.Text(
                        format_axis_value(value),
                        size=10,
                        color=ft.Colors.GREY_600,
                    ),
                )
                for value in axis_label_values(min_y, max_y)
            ],
        ),
        bottom_axis=fc.ChartAxis(
            title=ft.Text(x_axis_label, size=11, color=ft.Colors.GREY_700),
            title_size=28,
            show_labels=True,
            label_size=26,
            labels=[
                fc.ChartAxisLabel(
                    value=value,
                    label=ft.Text(
                        format_axis_value(value),
                        size=10,
                        color=ft.Colors.GREY_600,
                    ),
                )
                for value in axis_label_values(min_x, max_x)
            ],
        ),
    )
    return ft.Container(
        expand=True,
        padding=10,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=[
                ft.Text(title, weight=ft.FontWeight.BOLD),
                ft.Text(subtitle, size=12, color=ft.Colors.GREY_700),
                chart_legend(ft, series),
                chart,
            ],
            spacing=6,
        ),
    )


def chart_legend(
    ft: Any,
    series: tuple[tuple[str, tuple[tuple[float, float], ...], Any], ...],
) -> Any:
    if len(series) <= 1:
        return ft.Container(height=0)
    return ft.Row(
        controls=[
            ft.Text(label, size=12, color=color)
            for label, _, color in series
        ],
        spacing=12,
    )


def latest_training_metrics_table(
    ft: Any,
    metrics: tuple[dict[str, Any], ...],
) -> Any:
    rows = latest_training_metric_rows(metrics)
    if not rows:
        return ft.Text("No metrics logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Metric"),
            ft.DataColumn("Latest Value"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(metric),
                    ft.DataCell(value),
                ]
            )
            for metric, value in rows
        ],
    )


def best_checkpoint_card(ft: Any, run: DashboardRunSummary) -> Any:
    best = best_evaluation_checkpoint(run.evaluations, run.checkpoints)
    if best is None:
        return ft.Container(
            padding=12,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Text("No checkpoint evaluation score available yet."),
        )
    iteration, score, checkpoint_path = best
    return ft.Container(
        padding=12,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=[
                detail_row(ft, "Iteration", str(iteration)),
                detail_row(ft, "Mean RLlib Win", format_metric(score)),
                detail_row(ft, "Checkpoint", checkpoint_path or "n/a"),
            ],
            spacing=6,
        ),
    )


def checkpoints_table(
    ft: Any,
    run: DashboardRunSummary,
    *,
    players: int,
) -> Any:
    rows = checkpoint_table_rows(
        run.checkpoints,
        run.evaluations,
        run.latest_checkpoint_path,
    )
    if not rows:
        return ft.Text("No checkpoints logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Iteration"),
            ft.DataColumn("Created"),
            ft.DataColumn("Score"),
            ft.DataColumn("Markers"),
            ft.DataColumn("Path"),
            ft.DataColumn("Copy"),
            ft.DataColumn("Eval"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(iteration),
                    ft.DataCell(created_at),
                    ft.DataCell(score),
                    ft.DataCell(markers),
                    ft.DataCell(
                        ft.Text(path, selectable=True, size=12),
                    ),
                    ft.DataCell(
                        ft.IconButton(
                            icon=ft.Icons.CONTENT_COPY,
                            tooltip="Copy checkpoint path",
                            on_click=copy_text_handler(path),
                        )
                    ),
                    ft.DataCell(
                        ft.IconButton(
                            icon=ft.Icons.PLAY_ARROW,
                            tooltip="Copy evaluation command",
                            on_click=copy_text_handler(
                                checkpoint_evaluation_command(path, players)
                            ),
                        )
                    ),
                ]
            )
            for iteration, created_at, score, markers, path in rows
        ],
    )


def latest_evaluation_metrics_table(
    ft: Any,
    evaluations: tuple[dict[str, Any], ...],
) -> Any:
    rows = latest_evaluation_metric_rows(evaluations)
    if not rows:
        return ft.Text("No evaluation metrics logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Opponent"),
            ft.DataColumn("Episodes"),
            ft.DataColumn("RLlib Win"),
            ft.DataColumn("Opponent Win"),
            ft.DataColumn("Seat Spread"),
            ft.DataColumn("Illegal Rate"),
            ft.DataColumn("Combos"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(opponent),
                    ft.DataCell(evaluation_episodes),
                    ft.DataCell(rllib_win_rate),
                    ft.DataCell(opponent_win_rate),
                    ft.DataCell(seat_order_advantage),
                    ft.DataCell(illegal_action_rate),
                    ft.DataCell(combo_count),
                ]
            )
            for (
                opponent,
                evaluation_episodes,
                rllib_win_rate,
                opponent_win_rate,
                seat_order_advantage,
                illegal_action_rate,
                combo_count,
            ) in rows
        ],
    )


def combo_rate_table(
    ft: Any,
    evaluations: tuple[dict[str, Any], ...],
) -> Any:
    rows = combo_rate_rows(evaluations)
    if not rows:
        return ft.Text("No combo data logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Opponent"),
            ft.DataColumn("Combos / Episode"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(opponent),
                    ft.DataCell(combo_rate),
                ]
            )
            for opponent, combo_rate in rows
        ],
    )


def latest_action_distribution_table(
    ft: Any,
    evaluations: tuple[dict[str, Any], ...],
) -> Any:
    rows = latest_action_distribution_rows(evaluations)
    if not rows:
        return ft.Text("No action distribution logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Opponent"),
            ft.DataColumn("Category"),
            ft.DataColumn("Action"),
            ft.DataColumn("Count"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(opponent),
                    ft.DataCell(category),
                    ft.DataCell(action),
                    ft.DataCell(count),
                ]
            )
            for opponent, category, action, count in rows
        ],
    )


def latest_card_distribution_table(
    ft: Any,
    evaluations: tuple[dict[str, Any], ...],
) -> Any:
    rows = latest_card_distribution_rows(evaluations)
    if not rows:
        return ft.Text("No card distribution logged yet.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Opponent"),
            ft.DataColumn("Card"),
            ft.DataColumn("Count"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(opponent),
                    ft.DataCell(card),
                    ft.DataCell(count),
                ]
            )
            for opponent, card, count in rows
        ],
    )


def suspicious_behavior_panel(
    ft: Any,
    evaluations: tuple[dict[str, Any], ...],
) -> Any:
    messages = suspicious_behavior_messages(evaluations)
    if not messages:
        return ft.Container(
            padding=12,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Text("No suspicious behavior detected in the latest evaluations."),
        )
    return ft.Container(
        padding=12,
        border=border_all(ft, color=ft.Colors.AMBER_300),
        border_radius=8,
        content=ft.Column(
            controls=[
                ft.Text(message, color=ft.Colors.AMBER_900)
                for message in messages
            ],
            spacing=4,
        ),
    )


def config_summary_table(
    ft: Any,
    title: str,
    rows: tuple[tuple[str, str], ...],
) -> Any:
    return ft.Container(
        expand=True,
        padding=12,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=[
                ft.Text(title, weight=ft.FontWeight.BOLD),
                ft.DataTable(
                    columns=[
                        ft.DataColumn("Setting"),
                        ft.DataColumn("Value"),
                    ],
                    rows=[
                        ft.DataRow(
                            cells=[
                                ft.DataCell(label),
                                ft.DataCell(ft.Text(value, selectable=True)),
                            ]
                        )
                        for label, value in rows
                    ],
                ),
            ],
            spacing=8,
        ),
    )


def config_validation_panel(ft: Any, config: dict[str, Any]) -> Any:
    messages = config_validation_messages(config)
    if not messages:
        return ft.Container(
            padding=12,
            border=border_all(ft),
            border_radius=8,
            bgcolor=ft.Colors.WHITE,
            content=ft.Text("No config warnings detected."),
        )
    return ft.Container(
        padding=12,
        border=border_all(ft, color=ft.Colors.AMBER_300),
        border_radius=8,
        content=ft.Column(
            controls=[
                ft.Text(message, color=ft.Colors.AMBER_900)
                for message in messages
            ],
            spacing=4,
        ),
    )


def comparison_best_runs_table(
    ft: Any,
    runs: tuple[DashboardRunSummary, ...],
) -> Any:
    rows = comparison_best_run_rows(runs)
    return ft.DataTable(
        columns=[
            ft.DataColumn("Run"),
            ft.DataColumn("Latest Mean Win"),
            ft.DataColumn("Best Eval Score"),
            ft.DataColumn("Reward Mean"),
            ft.DataColumn("Env Steps/s"),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(run_name),
                    ft.DataCell(latest_mean_win),
                    ft.DataCell(best_eval_score),
                    ft.DataCell(reward_mean),
                    ft.DataCell(environment_steps_per_second),
                ]
            )
            for (
                run_name,
                latest_mean_win,
                best_eval_score,
                reward_mean,
                environment_steps_per_second,
            ) in rows
        ],
    )


def config_diff_table(
    ft: Any,
    runs: tuple[DashboardRunSummary, ...],
) -> Any:
    rows = config_diff_rows(runs)
    if not rows:
        return ft.Text("No differences found in the tracked config fields.")
    return ft.DataTable(
        columns=[
            ft.DataColumn("Setting"),
            *(ft.DataColumn(run.run_name) for run in runs),
        ],
        rows=[
            ft.DataRow(
                cells=[
                    ft.DataCell(setting),
                    *(ft.DataCell(ft.Text(value, selectable=True)) for value in values),
                ]
            )
            for setting, values in rows
        ],
    )


def copy_text_handler(text: str) -> Any:
    def handle_copy(event: Any) -> None:
        page = getattr(event.control, "page", None)
        if page is not None:
            page.clipboard.set_text(text)

    return handle_copy


def checkpoint_evaluation_command(checkpoint_path: str, players: int) -> str:
    seat_policies = [f"player_1=rllib:{checkpoint_path}"]
    for seat_number in range(2, players + 1):
        strategy = "random" if seat_number == 2 else "safe-rule"
        seat_policies.append(f"player_{seat_number}={strategy}")
    return (
        "venv\\Scripts\\python.exe scripts\\evaluate_rllib_checkpoint.py "
        f"--players {players} --episodes 100 "
        f"--seat-policies {','.join(seat_policies)} --wandb-mode disabled"
    )


def launcher_status_chip(ft: Any, status: str) -> Any:
    color = ft.Colors.BLUE_700
    background = ft.Colors.BLUE_50
    if status.startswith("Completed"):
        color = ft.Colors.GREEN_700
        background = ft.Colors.GREEN_50
    elif status.startswith("Exited") or status.startswith("Failed"):
        color = ft.Colors.RED_700
        background = ft.Colors.RED_50
    elif status.startswith("Stopping"):
        color = ft.Colors.AMBER_900
        background = ft.Colors.AMBER_50
    return ft.Container(
        padding=8,
        border_radius=8,
        bgcolor=background,
        content=ft.Text(status, size=12, color=color, weight=ft.FontWeight.BOLD),
    )


def stale_status_text(run: DashboardRunSummary) -> str:
    return f"{run.status} (stale)" if run.is_stale else run.status


def overview_metric_tile(ft: Any, label: str, value: str) -> Any:
    return ft.Container(
        expand=True,
        padding=9,
        border=border_all(ft),
        border_radius=8,
        bgcolor=ft.Colors.WHITE,
        content=ft.Column(
            controls=[
                ft.Text(label, size=12, color=ft.Colors.GREY_600),
                ft.Text(value, size=16, weight=ft.FontWeight.BOLD),
            ],
            spacing=2,
        ),
    )


def border_all(ft: Any, *, width: int = 1, color: Any | None = None) -> Any:
    side = ft.BorderSide(width=width, color=color or ft.Colors.GREY_200)
    return ft.Border(left=side, right=side, top=side, bottom=side)


def detail_row(ft: Any, label: str, value: str) -> Any:
    return ft.Row(
        controls=[
            ft.Text(label, width=150, color=ft.Colors.GREY_700),
            ft.Text(value, selectable=True, expand=True),
        ]
    )


def placeholder_tab(ft: Any, title: str, run: DashboardRunSummary | None) -> Any:
    if run is None:
        return empty_tab(ft)
    return ft.Container(
        padding=16,
        content=ft.Column(
            controls=[
                ft.Text(title, size=16, weight=ft.FontWeight.BOLD),
                ft.Text(
                    "This tab is scaffolded. The next phases will replace this "
                    "placeholder with simulator-specific charts and tables."
                ),
                ft.Text(f"Selected run: {run.run_name}"),
            ],
            spacing=8,
        ),
    )


def empty_tab(ft: Any) -> Any:
    return ft.Container(
        padding=16,
        content=ft.Text("No run selected. Start a training job or refresh the run list."),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the local Exploding Kittens experiment dashboard."
    )
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=DEFAULT_REFRESH_SECONDS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dashboard(
        experiment_dir=args.experiment_dir,
        refresh_seconds=args.refresh_seconds,
    )


if __name__ == "__main__":
    main()
