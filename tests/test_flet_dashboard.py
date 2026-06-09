from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from exploding_kittens.experiment_logging import LocalExperimentLogger
from scripts.dashboard_flet import (
    DashboardRunSummary,
    action_category,
    action_category_counts,
    best_evaluation_checkpoint,
    chart_bounds,
    checkpoint_for_iteration,
    checkpoint_table_rows,
    combo_rate_rows,
    comparison_best_run_rows,
    build_launcher_command,
    discover_config_files,
    display_path,
    format_command,
    axis_label_values,
    launched_run_selection,
    launcher_expected_run_dir,
    config_diff_rows,
    config_summary_rows,
    config_validation_messages,
    evaluation_points,
    evaluation_scores_by_iteration,
    format_metric,
    format_progress,
    format_duration_seconds,
    format_axis_value,
    is_stale_run,
    latest_action_distribution_rows,
    latest_card_distribution_rows,
    latest_evaluation_metric_rows,
    latest_evaluations_by_opponent,
    latest_win_rates,
    latest_training_metric_rows,
    launcher_duplicate_run_dir,
    launcher_run_dir,
    refresh_wait_seconds,
    load_dashboard_runs,
    metric_points,
    nested_config_value,
    parse_config_text,
    resolve_launcher_run_name,
    smooth_points,
    suspicious_behavior_messages,
    training_status_rows,
)


def test_load_dashboard_runs_discovers_and_summarizes_runs(tmp_path: Path) -> None:
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="run_a",
        config={},
    )
    logger.start()
    logger.log_metric(
        {
            "iteration": 3,
            "total_iterations": 10,
            "episode_reward_mean": 0.25,
            "episode_length_mean": 12.0,
            "environment_steps_per_second": 4.5,
            "elapsed_seconds": 90.0,
            "estimated_seconds_remaining": 210.0,
        }
    )
    logger.log_evaluation(
        {
            "iteration": 3,
            "opponent_strategy": "random",
            "rllib_combined_win_rate": 0.4,
        }
    )
    logger.complete(final_checkpoint_path=tmp_path / "checkpoint")

    runs = load_dashboard_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].run_name == "run_a"
    assert runs[0].status == "completed"
    assert runs[0].current_iteration == 3
    assert runs[0].total_iterations == 10
    assert runs[0].latest_reward_mean == 0.25
    assert runs[0].latest_episode_length_mean == 12.0
    assert runs[0].latest_environment_steps_per_second == 4.5
    assert runs[0].latest_win_rates == {"random": 0.4}
    assert runs[0].elapsed_seconds == 90.0
    assert runs[0].estimated_seconds_remaining == 210.0
    assert runs[0].active_config_name == "resolved_config.yaml"
    assert runs[0].checkpoints == ()
    assert str(tmp_path / "checkpoint") == runs[0].latest_checkpoint_path
    assert not runs[0].is_stale


def test_latest_win_rates_keeps_latest_value_per_opponent() -> None:
    win_rates = latest_win_rates(
        (
            {"opponent_strategy": "random", "rllib_combined_win_rate": 0.1},
            {"opponent_strategy": "safe-rule", "rllib_combined_win_rate": 0.2},
            {"opponent_strategy": "random", "rllib_combined_win_rate": 0.3},
        )
    )

    assert win_rates == {"random": 0.3, "safe-rule": 0.2}


def test_dashboard_formatters_handle_missing_values() -> None:
    run = load_dashboard_runs(Path("does-not-exist"))

    assert run == ()
    assert format_metric(None) == "n/a"
    assert format_metric(0.12345) == "0.123"
    assert format_duration_seconds(None) == "n/a"
    assert format_duration_seconds(65.0) == "01:05"
    assert format_duration_seconds(3661.0) == "1:01:01"


def test_format_progress_handles_missing_and_complete_values(tmp_path: Path) -> None:
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="progress",
        config={},
    )
    logger.start()
    no_metrics = load_dashboard_runs(tmp_path)[0]
    logger.log_metric({"iteration": 5, "total_iterations": 10})
    with_metrics = load_dashboard_runs(tmp_path)[0]

    assert format_progress(no_metrics) == "No iteration data yet"
    assert format_progress(with_metrics) == "5/10 (50.0%)"


def test_is_stale_run_only_flags_old_running_runs() -> None:
    now = datetime(2026, 6, 2, 12, 5, tzinfo=UTC)
    old = (now - timedelta(seconds=121)).isoformat()
    recent = (now - timedelta(seconds=30)).isoformat()

    assert is_stale_run(status="running", last_updated_at=old, now=now)
    assert not is_stale_run(status="running", last_updated_at=recent, now=now)
    assert not is_stale_run(status="completed", last_updated_at=old, now=now)
    assert is_stale_run(status="running", last_updated_at="not-a-date", now=now)


def test_metric_points_ignores_missing_metric_values() -> None:
    points = metric_points(
        (
            {"iteration": 1, "episode_reward_mean": 0.1},
            {"iteration": 2, "episode_reward_mean": None},
            {"iteration": 3},
            {"iteration": 4, "episode_reward_mean": 0.4},
        ),
        "episode_reward_mean",
    )

    assert points == ((1.0, 0.1), (4.0, 0.4))


def test_smooth_points_uses_trailing_window() -> None:
    smoothed = smooth_points(
        ((1.0, 1.0), (2.0, 3.0), (3.0, 5.0), (4.0, 7.0)),
        window=3,
    )

    assert smoothed == (
        (1.0, 1.0),
        (2.0, 2.0),
        (3.0, 3.0),
        (4.0, 5.0),
    )


def test_chart_bounds_adds_padding_and_handles_single_point() -> None:
    assert chart_bounds(()) == (0.0, 1.0, 0.0, 1.0)
    assert chart_bounds(((5.0, 10.0),)) == (5.0, 6.0, 9.0, 11.0)

    min_x, max_x, min_y, max_y = chart_bounds(((1.0, 10.0), (3.0, 20.0)))

    assert min_x == 1.0
    assert max_x == 3.0
    assert min_y == 9.0
    assert max_y == 21.0


def test_axis_labels_keep_charts_readable() -> None:
    assert axis_label_values(0.0, 1.0) == (0.0, 0.5, 1.0)
    assert format_axis_value(0.5) == "0.5"
    assert format_axis_value(10.0) == "10"
    assert format_axis_value(1200.0) == "1200"


def test_latest_training_metric_rows_formats_latest_metrics() -> None:
    rows = latest_training_metric_rows(
        (
            {"iteration": 1, "episode_reward_mean": 0.1},
            {
                "iteration": 2,
                "episode_reward_mean": 0.25,
                "episode_length_mean": 12.0,
                "environment_steps_sampled": 100,
                "agent_steps_sampled": 250,
                "environment_steps_per_second": 4.5,
                "elapsed_seconds": 65,
                "estimated_seconds_remaining": 130,
            },
        )
    )

    assert rows == (
        ("Iteration", "2"),
        ("Reward Mean", "0.250"),
        ("Episode Length Mean", "12.000"),
        ("Environment Steps", "100"),
        ("Agent Steps", "250"),
        ("Environment Steps/s", "4.5"),
        ("Elapsed", "01:05"),
        ("ETA", "02:10"),
    )


def test_evaluation_points_filters_by_opponent_and_metric() -> None:
    points = evaluation_points(
        (
            {
                "iteration": 10,
                "opponent_strategy": "random",
                "rllib_combined_win_rate": 0.3,
            },
            {
                "iteration": 10,
                "opponent_strategy": "safe-rule",
                "rllib_combined_win_rate": 0.2,
            },
            {
                "iteration": 20,
                "opponent_strategy": "random",
                "rllib_combined_win_rate": 0.5,
            },
        ),
        "random",
        "rllib_combined_win_rate",
    )

    assert points == ((10.0, 0.3), (20.0, 0.5))


def test_latest_evaluation_metric_rows_keeps_latest_opponent_values() -> None:
    rows = latest_evaluation_metric_rows(
        (
            {
                "iteration": 10,
                "opponent_strategy": "random",
                "evaluation_episodes": 5,
                "rllib_combined_win_rate": 0.3,
                "scripted_opponent_win_rate": 0.7,
                "seat_order_advantage": 0.2,
                "illegal_action_rate": 0.0,
                "combo_count": 1,
            },
            {
                "iteration": 20,
                "opponent_strategy": "random",
                "evaluation_episodes": 5,
                "rllib_combined_win_rate": 0.5,
                "scripted_opponent_win_rate": 0.5,
                "seat_order_advantage": 0.1,
                "illegal_action_rate": 0.1,
                "combo_count": 3,
            },
            {
                "iteration": 20,
                "opponent_strategy": "safe-rule",
                "evaluation_episodes": 5,
                "rllib_combined_win_rate": 0.25,
                "scripted_opponent_win_rate": 0.75,
                "seat_order_advantage": 0.4,
                "illegal_action_rate": 0.0,
                "combo_count": 2,
            },
        )
    )

    assert rows == (
        ("random", "5", "0.500", "0.500", "0.100", "0.100", "3"),
        ("safe-rule", "5", "0.250", "0.750", "0.400", "0.000", "2"),
    )


def test_latest_evaluations_by_opponent_keeps_latest_rows() -> None:
    latest = latest_evaluations_by_opponent(
        (
            {"iteration": 1, "opponent_strategy": "random", "score": 0.1},
            {"iteration": 2, "opponent_strategy": "random", "score": 0.2},
            {"iteration": 2, "opponent_strategy": "draw-only", "score": 0.3},
        )
    )

    assert latest["random"]["score"] == 0.2
    assert latest["draw-only"]["score"] == 0.3


def test_best_evaluation_checkpoint_matches_iteration_checkpoint() -> None:
    best = best_evaluation_checkpoint(
        (
            {
                "iteration": 10,
                "opponent_strategy": "random",
                "rllib_combined_win_rate": 0.4,
            },
            {
                "iteration": 10,
                "opponent_strategy": "safe-rule",
                "rllib_combined_win_rate": 0.2,
            },
            {
                "iteration": 20,
                "opponent_strategy": "random",
                "rllib_combined_win_rate": 0.5,
            },
            {
                "iteration": 20,
                "opponent_strategy": "safe-rule",
                "rllib_combined_win_rate": 0.7,
            },
        ),
        (
            {"iteration": 10, "checkpoint_path": "checkpoint_10"},
            {"iteration": 20, "checkpoint_path": "checkpoint_20"},
        ),
    )

    assert best == (20, 0.6, "checkpoint_20")


def test_checkpoint_for_iteration_uses_nearest_available_checkpoint() -> None:
    checkpoints = (
        {"iteration": 10, "checkpoint_path": "checkpoint_10"},
        {"iteration": 30, "checkpoint_path": "checkpoint_30"},
    )

    assert checkpoint_for_iteration(checkpoints, 10) == "checkpoint_10"
    assert checkpoint_for_iteration(checkpoints, 20) == "checkpoint_10"
    assert checkpoint_for_iteration(checkpoints, 5) == "checkpoint_10"


def test_checkpoint_table_rows_marks_latest_and_best() -> None:
    checkpoints = (
        {"iteration": 10, "checkpoint_path": "checkpoint_10", "created_at": "t1"},
        {"iteration": 20, "checkpoint_path": "checkpoint_20", "created_at": "t2"},
    )
    evaluations = (
        {
            "iteration": 10,
            "opponent_strategy": "random",
            "rllib_combined_win_rate": 0.2,
        },
        {
            "iteration": 20,
            "opponent_strategy": "random",
            "rllib_combined_win_rate": 0.8,
        },
    )

    assert evaluation_scores_by_iteration(evaluations) == {10: 0.2, 20: 0.8}
    assert checkpoint_table_rows(checkpoints, evaluations, "checkpoint_20") == (
        ("10", "t1", "0.200", "", "checkpoint_10"),
        ("20", "t2", "0.800", "latest, best", "checkpoint_20"),
    )


def test_action_categories_and_distribution_rows() -> None:
    evaluations = (
        {
            "iteration": 1,
            "opponent_strategy": "random",
            "action_distribution": {"draw": 8},
            "card_distribution": {},
        },
        {
            "iteration": 2,
            "opponent_strategy": "random",
            "action_distribution": {
                "draw": 2,
                "play_attack": 3,
                "play_favor": 1,
                "play_two_of_a_kind": 4,
            },
            "card_distribution": {"attack": 3, "favor": 1},
        },
    )

    assert action_category("draw") == "draw"
    assert action_category("play_attack") == "play card"
    assert action_category("play_favor") == "targeted card action"
    assert action_category("play_two_of_a_kind") == "combo action"
    assert action_category_counts(evaluations) == {
        "draw": 2,
        "play card": 3,
        "targeted card action": 1,
        "combo action": 4,
    }
    assert latest_action_distribution_rows(evaluations) == (
        ("random", "draw", "draw", "2"),
        ("random", "play card", "play_attack", "3"),
        ("random", "targeted card action", "play_favor", "1"),
        ("random", "combo action", "play_two_of_a_kind", "4"),
    )
    assert latest_card_distribution_rows(evaluations) == (
        ("random", "attack", "3"),
        ("random", "favor", "1"),
    )


def test_combo_rates_and_behavior_warnings() -> None:
    evaluations = (
        {
            "iteration": 1,
            "opponent_strategy": "random",
            "evaluation_episodes": 10,
            "combo_count": 5,
            "illegal_action_rate": 0.0,
            "action_distribution": {"draw": 9, "play_attack": 1},
        },
        {
            "iteration": 1,
            "opponent_strategy": "safe-rule",
            "evaluation_episodes": 10,
            "combo_count": 0,
            "illegal_action_rate": 0.2,
            "action_distribution": {"draw": 10},
        },
    )

    assert combo_rate_rows(evaluations) == (
        ("random", "0.500"),
        ("safe-rule", "0.000"),
    )
    assert suspicious_behavior_messages(evaluations) == (
        "High draw rate: latest evaluations are mostly draw actions.",
        "Illegal action rate against safe-rule: 0.200.",
    )


def test_config_helpers_parse_summarize_and_validate_yaml() -> None:
    config = parse_config_text(
        """
game:
  players: 3
  include_cards: [skip]
  exclude_cards: [favor]
training:
  iterations: 20
experiment:
  evaluation_interval: 5
  evaluation_episodes: 10
policy_setup:
  mode: shared
"""
    )

    assert nested_config_value(config, ("game", "players")) == 3
    assert config_summary_rows(
        config,
        (
            ("Players", ("game", "players")),
            ("Included", ("game", "include_cards")),
        ),
    ) == (("Players", "3"), ("Included", "skip"))
    assert config_validation_messages(config) == (
        "Both include_cards and exclude_cards are set.",
    )


def test_config_diff_rows_only_returns_changed_fields() -> None:
    run_a = make_dashboard_run(
        "a",
        resolved_config_text="""
game:
  players: 3
training:
  iterations: 10
experiment:
  evaluation_interval: 5
policy_setup:
  mode: shared
""",
    )
    run_b = make_dashboard_run(
        "b",
        resolved_config_text="""
game:
  players: 4
training:
  iterations: 10
experiment:
  evaluation_interval: 2
policy_setup:
  mode: shared
""",
    )

    assert config_diff_rows((run_a, run_b)) == (
        ("Players", ("3", "4")),
        ("Evaluation Interval", ("5", "2")),
    )


def test_comparison_best_run_rows_orders_by_latest_mean_win_rate() -> None:
    rows = comparison_best_run_rows(
        (
            make_dashboard_run(
                "low",
                latest_win_rates={"random": 0.2, "safe-rule": 0.4},
            ),
            make_dashboard_run(
                "high",
                latest_win_rates={"random": 0.8, "safe-rule": 0.6},
            ),
        )
    )

    assert rows[0][0] == "high"
    assert rows[0][1] == "0.700"


def test_launcher_helpers_build_command_and_resolve_run_name(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "smoke.yaml"
    config_path.write_text(
        """
experiment:
  run_name: config_run
dashboard_logging:
  run_name: dashboard_run
""",
        encoding="utf-8",
    )

    assert discover_config_files(config_dir) == (config_path,)
    assert resolve_launcher_run_name(
        config_path=config_path,
        run_name_override=None,
    ) == "dashboard_run"
    assert resolve_launcher_run_name(
        config_path=config_path,
        run_name_override="manual_run",
    ) == "manual_run"

    command = build_launcher_command(
        python_executable="python",
        config_path=config_path,
        experiment_dir=tmp_path / "experiments",
        run_name="manual_run",
        iterations="5",
        seed="11",
        wandb_mode="disabled",
        show_progress=False,
    )

    assert command[:3] == (
        "python",
        str(Path("C:/code/exploding_kittens/scripts/train_rllib_pettingzoo.py")),
        "--config",
    )
    assert "--dashboard-logging" in command
    assert "--experiment-dir" in command
    assert "--run-name" in command
    assert "--dashboard-run-name" in command
    assert "--iterations" in command
    assert "--seed" in command
    assert "--wandb-mode" in command
    assert "--no-progress" in command
    assert "manual_run" in format_command(command)


def test_launcher_duplicate_run_dir_detects_existing_directory(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  run_name: duplicate
""",
        encoding="utf-8",
    )
    experiment_dir = tmp_path / "experiments"
    expected_run_dir = launcher_run_dir(
        experiment_dir=experiment_dir,
        run_name="duplicate",
    )
    expected_run_dir.mkdir(parents=True)

    assert launcher_duplicate_run_dir(
        experiment_dir=experiment_dir,
        config_path=config_path,
        run_name_override=None,
    ) == expected_run_dir
    assert launcher_duplicate_run_dir(
        experiment_dir=experiment_dir,
        config_path=config_path,
        run_name_override="new_run",
    ) is None


def test_launcher_selection_and_refresh_helpers() -> None:
    selected_run, comparison_run_names = launched_run_selection(
        {"existing"},
        "new_training_run",
    )

    assert selected_run == "new_training_run"
    assert comparison_run_names == {"existing", "new_training_run"}
    assert refresh_wait_seconds(5, launcher_running=True) == 1
    assert refresh_wait_seconds(5, launcher_running=False) == 5


def test_launcher_expected_run_dir_and_training_status_rows(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
experiment:
  run_name: expected
""",
        encoding="utf-8",
    )
    expected_run_dir = launcher_expected_run_dir(
        experiment_dir=tmp_path / "experiments",
        config_path=config_path,
        run_name_override=None,
    )

    assert expected_run_dir == tmp_path / "experiments" / "expected"
    assert training_status_rows(
        None,
        {
            "status": "Running",
            "expected_run_dir": expected_run_dir,
            "resolved_run_name": "expected",
        },
    ) == (
        ("Launcher Status", "Running"),
        ("Expected Directory", str(expected_run_dir)),
        ("Metrics Rows", "0"),
        ("Last Metric Update", "n/a"),
        ("Selected Run", "expected"),
    )


def test_training_tab_launcher_text_fields_can_update_without_render() -> None:
    from types import SimpleNamespace

    import flet as ft

    calls: list[tuple[str, str, bool]] = []

    def set_field(field_name: str, value: str, render_now: bool = True) -> None:
        calls.append((field_name, value, render_now))

    from scripts.dashboard_flet import launcher_control_panel

    panel = launcher_control_panel(
        ft,
        {
            "config_files": (),
            "config_path": "configs/rllib_smoke.yaml",
            "run_name": "",
            "resolved_run_name": "smoke",
            "expected_run_dir": Path("reports/experiments/smoke"),
            "iterations": "",
            "seed": "",
            "wandb_mode": "use-config",
            "show_progress": True,
            "command_preview": "python train.py",
            "duplicate_run_dir": None,
            "output_lines": (),
            "status": "Idle",
            "is_running": False,
            "pending_stop": False,
        },
        {"set_field": set_field},
    )
    run_name_field = panel.content.controls[1].controls[1]
    event = SimpleNamespace(control=SimpleNamespace(value="abc"))

    run_name_field.on_change(event)
    run_name_field.on_blur(event)

    assert calls == [
        ("launcher_run_name", "abc", False),
        ("launcher_run_name", "abc", True),
    ]


def make_dashboard_run(
    run_name: str,
    *,
    latest_win_rates: dict[str, float] | None = None,
    resolved_config_text: str | None = None,
) -> DashboardRunSummary:
    return DashboardRunSummary(
        run_name=run_name,
        run_dir=Path(run_name),
        status="completed",
        current_iteration=None,
        total_iterations=None,
        latest_reward_mean=None,
        latest_episode_length_mean=None,
        latest_environment_steps_per_second=None,
        latest_win_rates=latest_win_rates or {},
        elapsed_seconds=None,
        estimated_seconds_remaining=None,
        latest_checkpoint_path=None,
        active_config_name=None,
        last_updated_at=None,
        is_stale=False,
        metrics=(),
        evaluations=(),
        checkpoints=(),
        resolved_config_text=resolved_config_text,
    )
