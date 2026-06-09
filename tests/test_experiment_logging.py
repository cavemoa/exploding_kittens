from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from exploding_kittens.experiment_logging import (
    CHECKPOINTS_FILENAME,
    COMPLETED_STATUS,
    EVALUATIONS_FILENAME,
    FAILED_STATUS,
    METRICS_FILENAME,
    RESOLVED_CONFIG_FILENAME,
    RUNNING_STATUS,
    SUMMARY_FILENAME,
    LocalExperimentLogger,
    discover_experiment_runs,
    read_experiment_run,
)


def test_local_experiment_logger_writes_and_reads_complete_run(tmp_path: Path) -> None:
    clock = FakeClock()
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="run_a",
        config={"training": {"iterations": 2}},
        clock=clock,
    )

    logger.start()
    logger.log_metric(
        {
            "iteration": 1,
            "total_iterations": 2,
            "environment_steps_sampled": 32,
        }
    )
    logger.log_evaluation(
        {
            "iteration": 1,
            "opponent_strategy": "random",
            "rllib_combined_win_rate": 0.5,
        }
    )
    logger.log_checkpoint(
        iteration=1,
        checkpoint_path=tmp_path / "models" / "checkpoint_000001",
    )
    logger.complete(final_checkpoint_path=tmp_path / "models" / "final")

    run = read_experiment_run(tmp_path / "run_a")

    assert run.summary["status"] == COMPLETED_STATUS
    assert run.metrics[0]["iteration"] == 1
    assert run.evaluations[0]["opponent_strategy"] == "random"
    assert run.checkpoints[0]["iteration"] == 1
    assert "iterations: 2" in (run.resolved_config_text or "")


def test_read_experiment_run_handles_empty_and_partial_logs(tmp_path: Path) -> None:
    empty = read_experiment_run(tmp_path / "missing")

    assert empty.summary["status"] == "unknown"
    assert empty.metrics == ()
    assert empty.evaluations == ()
    assert empty.checkpoints == ()
    assert empty.resolved_config_text is None

    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    (partial_dir / SUMMARY_FILENAME).write_text(
        '{"run_name": "partial", "status": "running"}',
        encoding="utf-8",
    )
    (partial_dir / METRICS_FILENAME).write_text(
        '{"iteration": 1}\n\n{"iteration": 2}\n',
        encoding="utf-8",
    )

    partial = read_experiment_run(partial_dir)

    assert partial.summary["status"] == RUNNING_STATUS
    assert [row["iteration"] for row in partial.metrics] == [1, 2]
    assert partial.evaluations == ()


def test_local_experiment_logger_marks_failed_run(tmp_path: Path) -> None:
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="broken",
        config={},
    )

    logger.start()
    logger.fail(RuntimeError("boom"))

    run = read_experiment_run(tmp_path / "broken")

    assert run.summary["status"] == FAILED_STATUS
    assert run.summary["error"]["type"] == "RuntimeError"
    assert run.summary["error"]["message"] == "boom"


def test_local_experiment_logger_appends_existing_run_logs(tmp_path: Path) -> None:
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="continued",
        config={"training": {"iterations": 100}},
    )
    logger.start()
    logger.log_metric({"iteration": 100})
    logger.complete(final_checkpoint_path=tmp_path / "models" / "checkpoint_000100")

    continued = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="continued",
        config={"training": {"iterations": 200}},
        append_existing=True,
    )
    continued.start()
    continued.log_metric({"iteration": 101})
    continued.set_wandb_run_id("abc123")

    run = read_experiment_run(tmp_path / "continued")

    assert [row["iteration"] for row in run.metrics] == [100, 101]
    assert run.summary["status"] == RUNNING_STATUS
    assert run.summary["final_checkpoint_path"] == str(
        tmp_path / "models" / "checkpoint_000100"
    )
    assert run.summary["wandb_run_id"] == "abc123"


def test_discover_experiment_runs_sorts_run_directories(tmp_path: Path) -> None:
    for run_name in ("z_run", "a_run"):
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        (run_dir / SUMMARY_FILENAME).write_text(
            f'{{"run_name": "{run_name}", "status": "completed"}}',
            encoding="utf-8",
        )

    runs = discover_experiment_runs(tmp_path)

    assert [run.summary["run_name"] for run in runs] == ["a_run", "z_run"]


def test_logger_creates_expected_dashboard_files(tmp_path: Path) -> None:
    logger = LocalExperimentLogger(
        enabled=True,
        experiment_dir=tmp_path,
        run_name="files",
        config={},
    )

    logger.start()

    run_dir = tmp_path / "files"
    assert (run_dir / METRICS_FILENAME).exists()
    assert (run_dir / EVALUATIONS_FILENAME).exists()
    assert (run_dir / CHECKPOINTS_FILENAME).exists()
    assert (run_dir / SUMMARY_FILENAME).exists()
    assert (run_dir / RESOLVED_CONFIG_FILENAME).exists()


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 6, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self.current
        self.current += timedelta(seconds=1)
        return value
