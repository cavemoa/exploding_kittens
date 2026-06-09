# Flet UI Implementation Plan

This plan builds a local-first, simulator-specific dashboard for Exploding Kittens training experiments. The goal is not to recreate WandB. The goal is to give us a clean offline interface that understands our runs, our configs, our checkpoints, our card/action diagnostics, and the questions we actually care about.

The first milestone is an MVP dashboard that can read local experiment logs and refresh while training is running. Later phases add richer charts, checkpoint inspection, run comparison, and eventually experiment launching.

## Phase 1: Define The Local Experiment Log

Testable outcome: every training run writes a simple local record that the dashboard can read without importing RLlib or starting training.

1. [x] Decide the run directory layout under `reports/experiments/`.
2. [x] Create a run directory per experiment, for example:
   - [x] `reports/experiments/rllib_shared_policy_medium/`
   - [x] `reports/experiments/rllib_shared_policy_medium/metrics.jsonl`
   - [x] `reports/experiments/rllib_shared_policy_medium/evaluations.jsonl`
   - [x] `reports/experiments/rllib_shared_policy_medium/checkpoints.jsonl`
   - [x] `reports/experiments/rllib_shared_policy_medium/summary.json`
   - [x] `reports/experiments/rllib_shared_policy_medium/resolved_config.yaml`
3. [x] Define the `metrics.jsonl` schema:
   - [x] iteration
   - [x] elapsed seconds
   - [x] estimated seconds remaining
   - [x] environment steps sampled
   - [x] agent steps sampled
   - [x] environment steps per second
   - [x] episode reward mean
   - [x] episode length mean
4. [x] Define the `evaluations.jsonl` schema:
   - [x] iteration
   - [x] opponent strategy
   - [x] RLlib combined win rate
   - [x] scripted opponent win rate
   - [x] seat-order advantage
   - [x] illegal action rate
   - [x] combo count
   - [x] action distribution
   - [x] card distribution
5. [x] Define the `checkpoints.jsonl` schema:
   - [x] iteration
   - [x] checkpoint path
   - [x] created timestamp
   - [x] best-known evaluation score if available
6. [x] Define the `summary.json` schema:
   - [x] run name
   - [x] status: `running`, `completed`, `failed`, or `unknown`
   - [x] started timestamp
   - [x] finished timestamp
   - [x] final checkpoint path
   - [x] final evaluation highlights
7. [x] Add tests for reading empty, partial, and complete experiment logs.

## Phase 2: Add Local Logging To RLlib Training

Testable outcome: `scripts/train_rllib_pettingzoo.py` writes local dashboard logs during training while keeping WandB optional.

1. [x] Add a `local_tracking` or `dashboard_logging` section to RLlib config files.
2. [x] Add config fields:
   - [x] `enabled`
   - [x] `experiment_dir`
   - [x] `run_name`
   - [x] `flush_each_iteration`
3. [x] Add CLI overrides:
   - [x] `--dashboard-logging`
   - [x] `--no-dashboard-logging`
   - [x] `--experiment-dir`
4. [x] Create a small local logging helper module.
5. [x] Write `resolved_config.yaml` to the experiment run directory.
6. [x] Append training iteration rows to `metrics.jsonl`.
7. [x] Append interval evaluation rows to `evaluations.jsonl`.
8. [x] Append saved checkpoint rows to `checkpoints.jsonl`.
9. [x] Write `summary.json` at run start.
10. [x] Update `summary.json` on completion.
11. [x] Mark `summary.json` as failed if training raises an exception.
12. [x] Add unit tests for log writing.
13. [x] Add a smoke test that runs one tiny training iteration and confirms dashboard log files exist.

## Phase 3: Scaffold The Flet Dashboard App

Testable outcome: a Flet app opens locally and shows a list of experiment runs from disk.

1. [x] Add optional UI dependencies to `pyproject.toml`:
   - [x] `flet`
   - [x] `flet-charts`
2. [x] Install the UI dependencies in the virtual environment.
3. [x] Create `scripts/dashboard_flet.py`.
4. [x] Add a run discovery function that scans `reports/experiments/`.
5. [x] Add a basic Flet window layout:
   - [x] left run selector
   - [x] top run title/status band
   - [x] main tab area
6. [x] Add tabs:
   - [x] Overview
   - [x] Training
   - [x] Evaluation
   - [x] Behavior
   - [x] Checkpoints
   - [x] Config
7. [x] Add a placeholder view for each tab.
8. [x] Add a refresh button.
9. [x] Add auto-refresh every few seconds.
10. [x] Add a command documented in the README:

```powershell
venv\Scripts\python.exe scripts\dashboard_flet.py
```

11. [x] Add tests for pure data loading functions.

## Phase 4: MVP Overview Tab

Testable outcome: the dashboard gives immediate feedback on whether a run is alive, how far through it is, and whether the agent is improving.

1. [x] Show run status: running, completed, failed, or unknown.
2. [x] Show active config name and run name.
3. [x] Show current iteration and total configured iterations.
4. [x] Show progress percentage.
5. [x] Show elapsed time.
6. [x] Show ETA.
7. [x] Show latest environment steps per second.
8. [x] Show latest episode reward mean.
9. [x] Show latest episode length mean.
10. [x] Show latest win rates against:
   - [x] random
   - [x] safe-rule
   - [x] draw-only
11. [x] Add a compact latest-checkpoint display.
12. [x] Add an empty-state view when no runs exist yet.
13. [x] Add a stale-run indicator if a running run has not updated recently.

## Phase 5: Training Tab

Testable outcome: training dynamics are visible over time without reading terminal output.

1. [x] Add win-rate charts against baseline opponents.
2. [x] Add illegal-action-rate chart by baseline opponent.
3. [x] Add environment steps sampled chart by iteration.
4. [x] Add environment steps per second chart by iteration.
5. [x] Add latest evaluation metrics table.
6. [x] Add latest training metrics table, including reward and episode length.
7. [x] Add chart empty states when metrics are not available yet.
8. [x] Use clean chart axis titles and value labels without overlap.
9. [x] Add a toggle for raw values versus smoothed values.
10. [x] Add tests for smoothing, evaluation grouping, and chart data preparation.

## Phase 6: Evaluation Tab

Testable outcome: baseline evaluation tells us whether training is beating simple opponents.

1. [x] Add win-rate line charts by opponent strategy.
2. [x] Show RLlib combined win rate against:
   - [x] random
   - [x] safe-rule
   - [x] draw-only
3. [x] Show scripted-opponent win rate for the same evaluations.
4. [x] Show illegal action rate over time.
5. [x] Show seat-order advantage over time.
6. [x] Show evaluation episode count.
7. [x] Add a latest evaluation table.
8. [x] Highlight best checkpoint by evaluation score.
9. [x] Add tests for grouping evaluation rows by opponent.

## Phase 7: Behavior Tab

Testable outcome: we can see what the model is actually doing, not just whether it wins.

1. [x] Show action distribution:
   - [x] draw
   - [x] play card
   - [x] targeted card action
   - [x] combo action
2. [x] Show cards played by type.
3. [x] Show combo count over time.
4. [x] Show latest combo rate per evaluation episode.
5. [x] Show card distribution by opponent evaluation.
6. [x] Highlight suspicious behavior:
   - [x] very high draw-only rate
   - [x] zero card play over many evaluations
   - [x] illegal action rate greater than zero
7. [x] Add tests for action/card distribution parsing.

## Phase 8: Checkpoints Tab

Testable outcome: saved checkpoints are easy to find, compare, and use for later evaluation.

1. [x] Show all known checkpoints for the selected run.
2. [x] Show checkpoint iteration.
3. [x] Show checkpoint path.
4. [x] Show creation timestamp.
5. [x] Show best available evaluation score.
6. [x] Mark the latest checkpoint.
7. [x] Mark the best checkpoint.
8. [x] Add a copy-path button for each checkpoint.
9. [x] Add a button to run evaluation for a selected checkpoint.
10. [x] Add tests for checkpoint ranking.

## Phase 9: Config Tab

Testable outcome: the exact settings used for a run are visible without opening files manually.

1. [x] Display `resolved_config.yaml` for the selected run.
2. [x] Show game settings:
   - [x] players
   - [x] max turns
   - [x] included cards
   - [x] excluded cards
   - [x] enabled combo rules
3. [x] Show training settings:
   - [x] iterations
   - [x] train batch size
   - [x] minibatch size
   - [x] rollout fragment length
   - [x] epochs
   - [x] learning rate
4. [x] Show experiment settings:
   - [x] checkpoint interval
   - [x] evaluation interval
   - [x] evaluation opponents
5. [x] Show policy setup:
   - [x] shared
   - [x] separate-per-seat
   - [x] learner-vs-frozen
   - [x] learner-vs-pool
6. [x] Add a config validation warning panel.
7. [x] Add tests for config summary extraction.

## Phase 10: Run Comparison

Testable outcome: two or more runs can be compared side by side.

1. [x] Add multi-select in the run selector.
2. [x] Add comparison charts for reward mean.
3. [x] Add comparison charts for win rate against each baseline.
4. [x] Add comparison charts for environment steps per second.
5. [x] Add a config-difference summary.
6. [x] Add a best-run table.
7. [x] Add tests for aligning runs with different iteration counts.

## Phase 11: Integrated Local Experiment Launcher

Testable outcome: experiments can be started from the Training tab, while the same tab follows local metrics and charts as the run writes dashboard logs.

1. [x] Add a config-file picker.
2. [x] Add editable run name.
3. [x] Add quick overrides:
   - [x] iterations
   - [x] seed
   - [x] WandB mode
   - [x] progress display
4. [x] Start training as a subprocess from the Training tab.
5. [x] Stream subprocess stdout into a Training tab log panel.
6. [x] Prevent starting duplicate runs with the same run directory.
7. [x] Add a stop button that asks for confirmation.
8. [x] Auto-select the launched run immediately on start.
9. [x] Show expected run directory and duplicate-run warnings inline.
10. [x] Show last metric update and metrics rows loaded.
11. [x] Refresh more quickly while a launched process is running.
12. [x] Add tests for command construction and refresh-state helpers.

## Phase 12: Polish And Packaging

Testable outcome: the dashboard feels like a useful local tool rather than a temporary script.

1. [x] Add a clear visual hierarchy and restrained dashboard styling.
2. [x] Add compact cards for key metrics.
3. [x] Add responsive layout for laptop screens.
4. [ ] Add keyboard-friendly refresh and navigation.
5. [ ] Add robust error messages for malformed logs.
6. [ ] Add README documentation.
7. [ ] Add screenshots or generated example data.
8. [ ] Add a small demo run fixture for UI development.
9. [ ] Consider packaging as a local desktop app.

## MVP Definition

The MVP is complete when:

1. [ ] RLlib training writes local `metrics.jsonl`, `evaluations.jsonl`, `checkpoints.jsonl`, `summary.json`, and `resolved_config.yaml`.
2. [ ] `scripts/dashboard_flet.py` starts without internet access.
3. [ ] The dashboard lists local runs.
4. [ ] The Overview tab shows progress, ETA, latest reward, and latest win rates.
5. [ ] The Training tab shows win-rate and training-speed charts.
6. [x] The Evaluation tab shows win-rate charts against baselines.
7. [x] The Config tab shows the exact YAML used for the selected run.
8. [x] The README explains how to start training and open the dashboard.
