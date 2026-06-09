from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import yaml


METRICS_FILENAME = "metrics.jsonl"
EVALUATIONS_FILENAME = "evaluations.jsonl"
CHECKPOINTS_FILENAME = "checkpoints.jsonl"
SUMMARY_FILENAME = "summary.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"
RUNNING_STATUS = "running"
COMPLETED_STATUS = "completed"
FAILED_STATUS = "failed"
UNKNOWN_STATUS = "unknown"


@dataclass(frozen=True)
class ExperimentRunData:
    run_dir: Path
    summary: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]
    evaluations: tuple[dict[str, Any], ...]
    checkpoints: tuple[dict[str, Any], ...]
    resolved_config_text: str | None


class LocalExperimentLogger:
    def __init__(
        self,
        *,
        enabled: bool,
        experiment_dir: Path,
        run_name: str,
        config: dict[str, Any],
        flush_each_iteration: bool = True,
        append_existing: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.enabled = enabled
        self.experiment_dir = experiment_dir
        self.run_name = run_name
        self.config = config
        self.flush_each_iteration = flush_each_iteration
        self.append_existing = append_existing
        self.clock = clock or utc_now
        self.run_dir = experiment_dir / run_name
        self.started_at: str | None = None

    @classmethod
    def disabled(cls) -> LocalExperimentLogger:
        return cls(
            enabled=False,
            experiment_dir=Path("reports/experiments"),
            run_name="disabled",
            config={},
        )

    def start(self) -> None:
        if not self.enabled:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            METRICS_FILENAME,
            EVALUATIONS_FILENAME,
            CHECKPOINTS_FILENAME,
        ):
            path = self.run_dir / filename
            if self.append_existing and path.exists():
                continue
            path.write_text("", encoding="utf-8")
        existing_summary = self._read_summary() if self.append_existing else {}
        self.started_at = existing_summary.get("started_at") or timestamp(self.clock)
        now = timestamp(self.clock)
        write_yaml(self.run_dir / RESOLVED_CONFIG_FILENAME, self.config)
        self._write_summary(
            {
                **existing_summary,
                "run_name": self.run_name,
                "status": RUNNING_STATUS,
                "started_at": self.started_at,
                "finished_at": None,
                "last_updated_at": now,
                "final_checkpoint_path": existing_summary.get("final_checkpoint_path"),
                "final_evaluation_highlights": existing_summary.get(
                    "final_evaluation_highlights",
                    {},
                ),
                "error": None,
            }
        )

    def set_wandb_run_id(self, wandb_run_id: str | None) -> None:
        if not self.enabled or not wandb_run_id:
            return
        current = self._read_summary()
        current["wandb_run_id"] = wandb_run_id
        current["last_updated_at"] = timestamp(self.clock)
        self._write_summary(current)

    def log_metric(self, row: dict[str, Any]) -> None:
        self._append_jsonl(METRICS_FILENAME, row)

    def log_evaluation(self, row: dict[str, Any]) -> None:
        self._append_jsonl(EVALUATIONS_FILENAME, row)

    def log_checkpoint(
        self,
        *,
        iteration: int,
        checkpoint_path: Path,
        best_known_evaluation_score: float | None = None,
    ) -> None:
        self._append_jsonl(
            CHECKPOINTS_FILENAME,
            {
                "iteration": iteration,
                "checkpoint_path": str(checkpoint_path),
                "created_at": timestamp(self.clock),
                "best_known_evaluation_score": best_known_evaluation_score,
            },
        )

    def complete(
        self,
        *,
        final_checkpoint_path: Path,
        final_evaluation_highlights: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        current = self._read_summary()
        finished_at = timestamp(self.clock)
        current.update(
            {
                "status": COMPLETED_STATUS,
                "finished_at": finished_at,
                "last_updated_at": finished_at,
                "final_checkpoint_path": str(final_checkpoint_path),
                "final_evaluation_highlights": final_evaluation_highlights or {},
                "error": None,
            }
        )
        self._write_summary(current)

    def fail(self, error: BaseException) -> None:
        if not self.enabled:
            return
        current = self._read_summary()
        failed_at = timestamp(self.clock)
        current.update(
            {
                "status": FAILED_STATUS,
                "finished_at": failed_at,
                "last_updated_at": failed_at,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        self._write_summary(current)

    def _append_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        if not self.enabled:
            return
        payload = dict(row)
        payload.setdefault("logged_at", timestamp(self.clock))
        with (self.run_dir / filename).open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")
            if self.flush_each_iteration:
                file.flush()

    def _read_summary(self) -> dict[str, Any]:
        path = self.run_dir / SUMMARY_FILENAME
        if not path.exists():
            return {
                "run_name": self.run_name,
                "status": UNKNOWN_STATUS,
                "started_at": self.started_at,
                "finished_at": None,
                "last_updated_at": None,
                "final_checkpoint_path": None,
                "final_evaluation_highlights": {},
                "error": None,
            }
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_summary(self, summary: dict[str, Any]) -> None:
        write_json(self.run_dir / SUMMARY_FILENAME, summary)


def read_experiment_run(run_dir: Path) -> ExperimentRunData:
    summary_path = run_dir / SUMMARY_FILENAME
    config_path = run_dir / RESOLVED_CONFIG_FILENAME
    return ExperimentRunData(
        run_dir=run_dir,
        summary=read_json(summary_path, default={"status": UNKNOWN_STATUS}),
        metrics=tuple(read_jsonl(run_dir / METRICS_FILENAME)),
        evaluations=tuple(read_jsonl(run_dir / EVALUATIONS_FILENAME)),
        checkpoints=tuple(read_jsonl(run_dir / CHECKPOINTS_FILENAME)),
        resolved_config_text=(
            config_path.read_text(encoding="utf-8") if config_path.exists() else None
        ),
    )


def discover_experiment_runs(experiment_dir: Path) -> tuple[ExperimentRunData, ...]:
    if not experiment_dir.exists():
        return ()
    run_dirs = tuple(path for path in experiment_dir.iterdir() if path.is_dir())
    return tuple(read_experiment_run(path) for path in sorted(run_dirs))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_json(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(default)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def utc_now() -> datetime:
    return datetime.now(UTC)


def timestamp(clock: Callable[[], datetime]) -> str:
    return clock().isoformat()
