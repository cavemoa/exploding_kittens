# Exploding Kittens Simulator

A small Python project for learning how to model a card game, test the rules, and later connect simple agents or reinforcement learning environments.

The first milestone is intentionally plain Python. PettingZoo can come later as a wrapper around the engine once the rules are stable.

## What Exists Now

- A simplified game engine in `src/exploding_kittens/engine.py`
- Card classes in `src/exploding_kittens/cards.py`
- Actions represented as an enum in `src/exploding_kittens/actions.py`
- Four simple agents:
  - `RandomAgent`
  - `DrawOnlyAgent`
  - `SkipIfPossibleAgent`
  - `SafeRuleAgent`
- A simulation script in `scripts/run_simulation.py`
- A focused pytest suite in `tests/test_engine.py`

## Simplified Rules

- The game supports two or more players.
- Each player starts with one `Defuse`.
- On a turn, the current player may draw or play `Skip` if they have one.
- Drawing a normal card adds it to the player's hand.
- Drawing an exploding kitten eliminates the player unless they have a `Defuse`.
- A used `Defuse` is discarded and the exploding kitten goes to the bottom of the draw pile.
- The game ends when one player remains.

## Architecture

The engine coordinates shared game flow: whose turn it is, drawing from the deck, discarding cards, advancing turns, and detecting the winner.

Agents choose concrete `GameAction` objects. A `GameAction` includes the action kind plus any choices needed to perform it, such as hand indexes and target player:

```python
GameAction(
    kind=Action.PLAY_FAVOR,
    card_indexes=(1,),
    target_player="player_3",
)
```

The `Action` enum is now the `GameAction.kind` value. The engine accepts `Action.DRAW` as a small debug convenience, but non-draw moves should always be submitted as full `GameAction` objects so there is no hidden target or card selection.

Agents receive a `GameObservation` instead of direct access to the full engine. The observation includes:

- exact names of the agent's own hand cards, in hand-index order
- own hand counts
- visible discard pile counts
- draw pile size
- number of living players
- current and next-player attack turn debt
- player-specific known top cards from `SeeTheFutureCard`
- public event kinds
- legal concrete actions

This creates a clean boundary between full game state and what an agent is allowed to use for decisions.

Agents can inspect which own-hand cards an action would spend:

```python
for action, card_names in observation.legal_actions_with_card_names():
    ...
```

For example, a Two of a Kind action with `card_indexes=(2, 3)` can be mapped back to `("skip", "skip")` before the agent decides whether that pair is worth spending.

For reinforcement-learning wrappers, `ActionEncoder` maps concrete `GameAction` objects to a stable integer action space and produces action masks:

```python
encoder = ActionEncoder(max_hand_size=32, max_players=5)
action_id = encoder.encode(action, observation)
game_action = encoder.decode(action_id, observation)
mask = encoder.action_mask(observation)
```

The current integer layout supports draw actions, single-card hand-slot actions, targeted single-card actions, targeted two-card combo actions, and a reserved range for later named-card choices.

`ObservationEncoder` converts `GameObservation` into fixed-shape numeric arrays:

```python
encoder = ObservationEncoder(max_hand_size=32, max_players=5)
encoded = encoder.encode(observation)
vector = encoded["observation"]
mask = encoded["action_mask"]
```

The vector includes own-hand slots, own hand counts, discard counts, draw pile size, living player count, attack turn debt, and known top cards from `SeeTheFutureCard`. Opponent hand sizes are included only when the encoder is configured with `include_opponent_card_counts=True`.

Hidden information stays hidden from observations:

- draw pile order
- exact cards in other players' hands

Opponent card counts are hidden by default. They can be revealed as simple hand sizes by setting:

```yaml
reveal_opponent_card_counts: true
```

Each card class owns its local rule effect:

- `NormalCard.on_draw()` adds the card to the player's hand.
- `SkipCard.play()` discards the skip card and ends the player's turn without drawing.
- `AttackCard.play()` discards the attack card and makes the next player take two turns.
- `ShuffleCard.play()` discards the shuffle card and shuffles the draw pile.
- `FavorCard.play()` targets the next living player with cards and takes one random card from their hand.
- `AlterTheFutureCard.play()` reverses the top three draw-pile cards. This is a custom learning card for this simulator, not a card from the commercial game.
- `SeeTheFutureCard.play()` stores the current top three draw-pile card names in that player's observations until the draw pile changes.
- `ExplodingKittenCard.on_draw()` either eliminates the player or asks a `DefuseCard` to save them.
- `DefuseCard.use_to_defuse()` discards the defuse and puts the exploding kitten back into the draw pile.

This keeps new card rules close to the card class while leaving global game state in the engine.

## Setup

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run Tests

```powershell
venv\Scripts\python.exe -m pytest
```

## Run A Simulation

The runner automatically reads `game_config.yaml` when it exists:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py
```

Current config options:

```yaml
games: 1000
players: 2
strategy: random
seed: 1
max_turns: 500
verbose: false
reveal_opponent_card_counts: false
include_cards:
exclude_cards: []
enabled_combo_rules: []
```

Command-line arguments override the YAML values:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --games 100 --players 3 --strategy random
```

Optional cards can be selected for focused experiments:

```yaml
include_cards:
  - normal
  - skip
exclude_cards:
  - attack
```

Valid optional card names are `normal`, `skip`, `attack`, `shuffle`, `favor`, `alter_the_future`, and `see_the_future`. `defuse` and `exploding_kitten` are always included because the current game-ending rules depend on them.

`alter_the_future` is a custom learning card for this simulator. It is included so we can experiment with inspecting and reordering the draw pile before implementing more advanced agent choices.

The same card filters can be overridden on the command line with comma-separated values:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --include-cards normal,skip --exclude-cards attack
```

Combo rules are separate from individual card classes. They look for patterns in a player's hand and can be enabled independently:

```yaml
enabled_combo_rules:
  - two_of_a_kind
```

Known combo rules:

- `two_of_a_kind`: play any two cards with the same title, discard both, then steal one random card from the next living player with cards.

Internally, combo rules also generate concrete `GameAction` choices. For example, Two of a Kind specifies which two hand indexes are being played and which player is targeted.

Combo rules can also be enabled from the command line:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --enabled-combo-rules two_of_a_kind
```

The simulation output includes diagnostics that help spot rule bugs:

- finished games vs games that hit `max_turns`
- average turns across all games
- average turns to win among completed games
- shortest and longest game length
- draws, skips, attacks, shuffles, favors, Alter the Future plays, See the Future plays, defuses, and eliminations per game
- combo rule plays such as Two of a Kind
- per-agent ASCII tables for win rate, average turns survived, cards played by type, combos played, explosions, defuses, and average hand size

Try the other baseline strategies:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --games 1000 --strategy draw-only
venv\Scripts\python.exe scripts\run_simulation.py --games 1000 --strategy skip-if-possible
venv\Scripts\python.exe scripts\run_simulation.py --games 1000 --strategy safe-rule
```

## PettingZoo Wrapper

The simulator can also run as a PettingZoo AEC environment:

```python
from exploding_kittens import pettingzoo_env

env = pettingzoo_env(players=3, enabled_combo_rules=("two_of_a_kind",))
env.reset(seed=1)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    if termination or truncation:
        action = None
    else:
        action = env.action_space(agent).sample(observation["action_mask"])
    env.step(action)
```

The PettingZoo observation is a dictionary containing a numeric `observation` vector and an `action_mask`. The wrapper uses `GameEngine` for all rule logic and `ActionEncoder` / `ObservationEncoder` for the RL-facing interface.

The wrapper starts with sparse rewards:

- winner: `+1`
- eliminated player: `-1`
- max-turn truncation: `0`

Each agent's cumulative total for the current episode is available in `info["episode_reward"]`.

The PettingZoo/RLlib environment can also use terminal rank rewards. This delays reward until a real winner exists, then assigns rewards by final placement based on elimination order. For a 4-player game, a common profile is:

```yaml
reward:
  profile: terminal-rank
  terminal_rank_rewards: [1.0, 0.3, -0.3, -1.0]
```

The list is ordered from 1st place to last place. `profile: sparse` keeps the original winner/elimination rewards.

Run a random masked PettingZoo rollout:

```powershell
venv\Scripts\python.exe scripts\run_pettingzoo_random.py --players 3 --seed 4 --enabled-combo-rules two_of_a_kind
```

## Single-Agent Gymnasium Wrapper

For single-policy reinforcement learning experiments, the project also includes a Gymnasium wrapper that trains one learner seat against scripted opponents:

```python
from exploding_kittens import SingleAgentEnv

env = SingleAgentEnv(
    players=3,
    learner="player_1",
    opponent_strategy="safe-rule",
    enabled_combo_rules=("two_of_a_kind",),
)

observation, info = env.reset(seed=1)
action = env.action_space.sample(observation["action_mask"])
observation, reward, terminated, truncated, info = env.step(action)
```

The wrapper exposes only the learner's encoded observation and action mask. Scripted opponents act automatically until it is the learner's turn again or the episode ends.

Supported scripted opponent strategies are:

- `random`
- `draw-only`
- `safe-rule`
- `skip-if-possible`

Rewards match the current sparse training setup from the PettingZoo wrapper:

- learner wins: `+1`
- learner is eliminated: `-1`
- max-turn truncation: `0`

## WandB Experiment Tracking

Weights & Biases is optional and is used for experiment tracking rather than core gameplay. Install the tracking extra when you want to log evaluation runs:

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev,tracking]"
```

Run a local evaluation without WandB:

```powershell
venv\Scripts\python.exe scripts\evaluate_single_agent.py --episodes 10 --players 3 --opponent-strategy safe-rule --wandb-mode disabled
```

Run the same evaluation with offline WandB logging:

```powershell
venv\Scripts\python.exe scripts\evaluate_single_agent.py --episodes 10 --players 3 --opponent-strategy safe-rule --wandb-mode offline
```

Use `--wandb-mode online` after `wandb login` to sync runs to the WandB dashboard.

The evaluator currently uses a random masked learner policy. It logs aggregate metrics such as win rate, average reward, average turns, and truncation rate. It also logs per-episode metrics and, for the first few episodes, a trace table with action IDs, action kinds, rewards, turn counts, visible game state, and known top cards.

## Mask-Aware PPO Training

The first training loop uses `sb3-contrib` `MaskablePPO` so the learner respects legal action masks instead of wasting updates on impossible moves. Install the training extra first:

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev,tracking,training]"
```

Run a tiny local smoke train against draw-only opponents:

```powershell
venv\Scripts\python.exe scripts\train_single_agent.py --total-timesteps 64 --players 2 --opponent-strategy draw-only --wandb-mode disabled
```

Run the same style of training with offline WandB logging:

```powershell
venv\Scripts\python.exe scripts\train_single_agent.py --total-timesteps 64 --players 2 --opponent-strategy draw-only --wandb-mode offline
```

Models are saved under `models/`, which is ignored by Git. WandB training runs log the configuration, `train/episode_reward`, `train/episode_length`, and final evaluation metrics such as `eval/win_rate`, `eval/average_reward`, and `eval/average_turns`.

Evaluate a saved MaskablePPO model against several scripted opponents:

```powershell
venv\Scripts\python.exe scripts\evaluate_agent.py --model-path models\maskable_ppo_single_agent_seed_1.zip --episodes 10 --opponent-strategies draw-only,random,safe-rule --wandb-mode disabled
```

Use `--wandb-mode offline` or `--wandb-mode online` to log the saved-policy comparison. The evaluator reports win rate, average reward, average turns survived, truncation rate, defuses, explosions, and cards played by type for each opponent strategy.

Run a complete visible training experiment in one command:

```powershell
venv\Scripts\python.exe scripts\run_training_experiment.py --total-timesteps 5000 --episodes 100 --training-opponent-strategy draw-only --opponent-strategies draw-only,random,safe-rule --wandb-mode offline
```

This trains a MaskablePPO model, evaluates it against the listed opponents, evaluates a random masked learner against the same opponents, prints a side-by-side comparison table, logs comparison metrics to WandB, and writes a JSON summary under `reports/`.

## Multi-Policy 3-Player Evaluation

For multi-seat experiments, train 3-player-compatible policies first:

```powershell
venv\Scripts\python.exe scripts\train_single_agent.py --players 3 --opponent-strategy random --total-timesteps 5000 --run-name ppo_3p_a --wandb-mode offline
venv\Scripts\python.exe scripts\train_single_agent.py --players 3 --opponent-strategy random --total-timesteps 5000 --run-name ppo_3p_b --wandb-mode offline
```

Then evaluate two trained seats against one random baseline:

```powershell
venv\Scripts\python.exe scripts\evaluate_multi_policy_game.py --players 3 --episodes 100 --seat-policies player_1=model:models\ppo_3p_a_seed_1.zip,player_2=model:models\ppo_3p_b_seed_1.zip,player_3=random --wandb-mode offline
```

The multi-policy evaluator reports per-seat win rate, average reward, average turns survived, defuses, explosions, cards played by type, combined trained-agent win rate, and random-agent win rate. It can also log a `multi_policy/seat_comparison` table to WandB.

## RLlib Shared-Policy Smoke Training

RLlib is the first multi-agent training framework we are trying. Install the multi-agent extra when you want to run it:

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev,tracking,training,multiagent]"
```

Run a tiny 3-player shared-policy PPO smoke job:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --iterations 1 --players 3 --max-turns 50 --train-batch-size 32 --minibatch-size 16 --rollout-fragment-length 16 --num-epochs 1 --wandb-mode disabled
```

Or run the same kind of job from a versionable YAML config:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_smoke.yaml
```

Command-line values override the YAML file when supplied:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_smoke.yaml --run-name smoke_override --seed 7 --wandb-mode offline
```

There are four starter RLlib configs:

- `configs/rllib_smoke.yaml`: one tiny local plumbing run
- `configs/rllib_medium.yaml`: a modest offline WandB run with `two_of_a_kind`, checkpoint intervals, and baseline evaluation intervals
- `configs/rllib_longer.yaml`: a longer template for more serious shared-policy experiments
- `configs/rllib_separate_per_seat_4p_longer.yaml`: a longer 4-player run with one trainable policy per player seat

The script wraps the existing PettingZoo environment with RLlib's `PettingZooEnv`, maps every seat to one shared policy, and uses Ray's old-API `TorchActionMaskModel` so illegal actions remain masked during training. Checkpoints are saved under `models/rllib/`, which is ignored by Git.

Each checkpoint directory also gets a `resolved_config.yaml` file so the run can be repeated later. Use `--wandb-mode offline` or `--wandb-mode online` to log RLlib smoke metrics such as sampled environment steps, sampled agent steps, mean episode reward, and mean episode length.

RLlib training shows a compact TQDM progress bar by default, including elapsed time, ETA, sampled environment steps, sampled agent steps, steps per second, mean reward, and mean episode length. Evaluation intervals print a concise win-rate summary instead of printing every iteration as a separate line. The starter YAML configs also suppress known noisy Ray/RLlib warnings by default. Use `--no-progress` to hide the local progress output, or `--show-ray-warnings` if you want to see the library warnings while debugging.

The starter RLlib configs also write local dashboard logs under `reports/experiments/<run_name>/`. Each run directory contains `metrics.jsonl`, `evaluations.jsonl`, `checkpoints.jsonl`, `summary.json`, and `resolved_config.yaml`; these files are intended to feed the local Flet dashboard. Use `--no-dashboard-logging` to disable this, `--dashboard-logging` to force it on, or `--experiment-dir reports\my_experiments` to write somewhere else.

If the configured `run_name` already has checkpoints or dashboard logs, the trainer pauses before starting Ray and asks whether to continue, overwrite, or quit. Continue restores from the latest numbered checkpoint. By default, `continue_mode: new-run` writes to a new continuation run such as `my_run_continue_001`, preserving the original run. Use `continue_mode: same-run` to keep the same run name, append local dashboard logs, continue checkpoint numbering, and resume the stored WandB run id when one is available from the local summary. Overwrite requires typing `OVERWRITE <run_name>` before existing logs/checkpoints are deleted. For automation, use `--existing-run-action continue`, `--existing-run-action overwrite`, or `--existing-run-action quit`; `ask` is the default. You can also set `--continue-mode same-run` from the command line. Explicit manual resume still works with `--restore-checkpoint models\rllib\old_run\checkpoint_000100 --run-name old_run_continue_manual`.

Open the local Flet dashboard:

```powershell
venv\Scripts\python.exe scripts\dashboard_flet.py
```

The dashboard lists runs from `reports/experiments/`, lets you refresh manually, and auto-refreshes in the background. The Overview tab shows run status, stale-run state, progress, elapsed time, ETA, latest reward, episode length, environment steps per second, latest baseline win rates, config file, checkpoint path, and run directory. The Training tab is the live control surface: it can start a local RLlib process from a YAML config, apply quick overrides, auto-select the launched run, stream output, prevent duplicate run directories, stop a run after confirmation, and show last metric update plus metrics rows loaded above the training charts. Charts show compact X/Y value labels plus axis titles. The Evaluation tab focuses on baseline results, scripted-opponent win rates, illegal action rate, seat-order advantage, evaluation episode count, and the best checkpoint by mean baseline score. The Behavior tab shows action buckets, cards played, combo trends, combo rate, and suspicious-behavior warnings. The Checkpoints tab lists checkpoint paths, scores, latest/best markers, and copy buttons for paths or evaluation commands. The Config tab shows the resolved YAML plus compact game, training, experiment, and policy summaries. The Comparison tab uses the checkboxes in the left run list to compare reward, win rate, speed, best scores, and key config differences across runs. Use `--experiment-dir reports\my_experiments` to point it at another local run directory.

Run the medium shared-policy loop:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_medium.yaml
```

The medium and longer configs save interval checkpoints such as `checkpoint_000010`, evaluate the current checkpoint against `random`, `safe-rule`, and `draw-only`, and log those metrics separately from PPO training. `evaluation_scope: primary-seat` evaluates the `player_1` policy and logs under `rllib_eval_during_training/...`. `evaluation_scope: native-seat` evaluates each trainable policy in its own seat and logs per-player metrics such as `eval_by_player/random/player_1/win_rate`, `eval_by_player/safe-rule/player_2/win_rate`, and `eval_by_player/draw-only/player_3/win_rate`; these are useful for WandB charts with one chart per opponent and one line per player. `evaluation_scope: randomized-seat-rank` evaluates each trainable policy as a general card player: it balances the policy evenly across the physical seats, fills the other seats with a random selection from `evaluation_opponents`, and scores each game with the configured terminal-rank rewards. For a 4-player config with `evaluation_episodes: 100`, each policy plays 25 games as each seat. It logs generalized metrics such as `general_eval/player_1_policy/win_rate`, `general_eval/player_1_policy/average_rank_reward`, `general_eval/player_1_policy/average_finish_position`, and per-seat win rates under `general_eval/player_1_policy/seat_win_rate/player_1`. The training evaluator tracks RLlib win rate, scripted-opponent win rate, seat-order advantage, illegal action rate, action distribution, cards played by type, combo count, or generalized rank metrics depending on the selected scope. It also compares the final checkpoint against the first checkpoint with `rllib_eval_final_vs_first/...` metrics.

RLlib configs can choose the training reward profile with a top-level `reward` section. `sparse` is the original profile; `terminal-rank` uses final placement rewards and stores the exact weights in the run's `resolved_config.yaml`.

The RLlib action-mask model currently uses a fully connected neural network inside RLlib's `TorchActionMaskModel`. By default it has two hidden layers, `[256, 256]`, with `tanh` activation. You can change this with a top-level `model` section:

```yaml
model:
  hidden_layers: [512, 256]
  activation: relu
```

The same values can be overridden from the command line:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_medium.yaml --model-hidden-layers 512,256 --model-activation relu
```

Changing `model.hidden_layers` or `model.activation` changes checkpoint compatibility, so the existing-run guard treats model changes as incompatible with continuing from an older checkpoint.

For a quick local proof of that loop, override the config down to two iterations:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_smoke.yaml --run-name serious_loop_smoke --iterations 2 --checkpoint-interval 1 --evaluation-interval 1 --evaluation-episodes 1 --evaluation-opponents random,safe-rule,draw-only --wandb-mode disabled
```

Evaluate a saved RLlib checkpoint against scripted opponents:

```powershell
venv\Scripts\python.exe scripts\evaluate_rllib_checkpoint.py --players 3 --episodes 100 --seat-policies player_1=rllib:models\rllib\smoke_rllib_shared_policy,player_2=random,player_3=safe-rule --wandb-mode offline
```

The RLlib checkpoint evaluator loads the saved shared policy, assigns policies by seat, runs deterministic seeded games, and reports per-seat win rate, average reward, average turns survived, explosions, defuses, cards played by type, RLlib-controlled win rate, scripted-opponent win rate, and seat-order advantage. Use `--wandb-mode disabled` for local console-only evaluation.

Build policy-pool metadata from a run directory and preview sampled opponents:

```powershell
venv\Scripts\python.exe scripts\build_policy_pool.py --checkpoint-dir models\rllib\serious_loop_smoke --metadata-path reports\policy_pool_smoke.json --players 3 --sample-mode mixed --seed 4
```

The policy pool includes scripted baselines (`random`, `safe-rule`, `draw-only`) plus discovered RLlib checkpoints. It labels checkpoint entries as `latest`, `previous`, and `best` when scores are supplied through `--scores`. Sampling modes are:

- `latest-only`: every opponent seat uses the latest checkpoint
- `random-historical`: opponent seats are sampled from RLlib checkpoint entries
- `mixed`: opponent seats are sampled from both scripted policies and checkpoints

This pool is the staging piece for the next phase, where frozen checkpoint policies can become actual training opponents.

The RLlib trainer now supports a `policy_setup` section. `shared` remains the default:

```yaml
policy_setup:
  mode: shared
```

Use separate trainable policies per seat with:

```yaml
policy_setup:
  mode: separate-per-seat
```

Run the tiny separate-per-seat smoke config:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_separate_per_seat_smoke.yaml
```

Run the longer 4-player separate-per-seat config:

```powershell
venv\Scripts\python.exe scripts\train_rllib_pettingzoo.py --config configs\rllib_separate_per_seat_4p_longer.yaml
```

Use frozen checkpoint opponents with:

```yaml
policy_setup:
  mode: learner-vs-frozen
  seat_policies:
    player_1: learner_policy
    player_2: player_2_frozen_policy
    player_3: player_3_frozen_policy
  trainable_policies:
    - learner_policy
  frozen_policies:
    player_2_frozen_policy:
      checkpoint_path: models/rllib/smoke_rllib_shared_policy
      policy_id: shared_policy
    player_3_frozen_policy:
      checkpoint_path: models/rllib/smoke_rllib_shared_policy
      policy_id: shared_policy
```

Use policy-pool frozen opponents with:

```yaml
policy_setup:
  mode: learner-vs-pool
  trainable_policies:
    - learner_policy
  opponent_pool:
    metadata_path: reports/policy_pool_smoke.json
    sample_mode: latest-only
    seed: 1
```

Train-time policy-pool sampling currently requires RLlib checkpoint opponents, so use `latest-only` or `random-historical`. `mixed` remains useful for previewing/evaluation pools, but scripted entries are not yet loaded as RLlib training policies.

## Good Next Steps

1. Add more deterministic rule tests.
2. Add more card rules one at a time.
3. Add an observation/action-mask layer.
4. Teach simple agents to use `known_top_cards`.
5. Wrap the stable engine in PettingZoo.
