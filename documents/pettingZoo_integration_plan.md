# PettingZoo Integration Plan

This plan breaks the PettingZoo integration into testable phases. The goal is to keep the current `GameEngine` as the source of game rules, then add wrappers and encoders around it so standard reinforcement learning tools can use the simulator.

## Phase 1: Stabilize The Agent Boundary

Testable outcome: existing agents can choose meaningful legal actions using only `GameObservation`, with no direct access to `GameEngine`.

1. [x] Add `GameObservation.own_hand` so agents can inspect their own exact hand in hand-index order.
2. [x] Add helpers for mapping `GameAction.card_indexes` back to card names.
3. [x] Keep hidden information out of observations:
   - [x] no draw pile order
   - [x] no exact opponent hands
   - [x] opponent hand sizes hidden by default
4. [x] Add a `SafeRuleAgent` that proves the observation shape is usable.
5. [x] Add tests showing `SafeRuleAgent` can:
   - [x] avoid drawing when `known_top_cards[0] == "exploding_kitten"`
   - [x] prefer `shuffle`, `skip`, `attack`, or `alter_the_future` when drawing is dangerous
   - [x] avoid spending `defuse` in `TwoOfAKindRule`
   - [x] prefer low-value pairs such as `normal`, `normal`

## Phase 2: Define A Stable Integer Action Space

Testable outcome: every legal `GameAction` can be represented by a stable integer action ID, and illegal action IDs can be masked out.

1. [x] Create `src/exploding_kittens/action_encoding.py`.
2. [x] Define an `ActionEncoder` class.
3. [x] Define a stable action ID layout, for example:
   - [x] draw
   - [x] play one card from hand slot
   - [x] play targeted card from hand slot against target player
   - [x] play two-card combo from hand slots against target player
   - [x] reserve space for future choices such as named-card requests
4. [x] Decide maximum supported hand size for the first RL wrapper: default `max_hand_size=32`.
5. [x] Decide maximum supported player count for the first RL wrapper: default `max_players=5`.
6. [x] Implement `encode(action: GameAction, observation: GameObservation) -> int`.
7. [x] Implement `decode(action_id: int, observation: GameObservation) -> GameAction`.
8. [x] Implement `action_mask(observation: GameObservation) -> np.ndarray`.
9. [x] Add tests that:
   - [x] every current legal action maps to exactly one valid action ID
   - [x] every valid action ID decodes back to the expected `GameAction`
   - [x] illegal action IDs are masked out
   - [x] target-specific actions preserve the selected target player
   - [x] combo actions preserve the selected card indexes

## Phase 3: Define A Numeric Observation Encoder

Testable outcome: `GameObservation` can be converted into fixed-shape numeric arrays suitable for Gymnasium, PettingZoo, and RL libraries.

1. [x] Create `src/exploding_kittens/observation_encoding.py`.
2. [x] Define an `ObservationEncoder` class.
3. [x] Encode own hand by fixed hand slots.
4. [x] Encode own hand counts by card type.
5. [x] Encode discard pile counts by card type.
6. [x] Encode draw pile size.
7. [x] Encode living player count.
8. [x] Encode current attack turn debt.
9. [x] Encode next-player attack turn debt.
10. [x] Encode `known_top_cards` from `SeeTheFutureCard`.
11. [x] Encode opponent hand sizes only when configured.
12. [x] Return an observation dictionary like:

```python
{
    "observation": np.ndarray,
    "action_mask": np.ndarray,
}
```

13. [x] Add tests that:
   - [x] encoded observations have stable shapes
   - [x] encoded observations do not expose draw pile order
   - [x] encoded observations do not expose exact opponent hands
   - [x] `known_top_cards` appears only after `SeeTheFutureCard`
   - [x] action masks match `engine.legal_actions()`

## Phase 4: Add The PettingZoo AEC Environment

Testable outcome: the simulator can run as a valid PettingZoo AEC environment.

1. [x] Add PettingZoo and Gymnasium to project dependencies.
2. [x] Create `src/exploding_kittens/pettingzoo_env.py`.
3. [x] Add an `env()` factory function.
4. [x] Add a `raw_env` class inheriting from `pettingzoo.AECEnv`.
5. [x] Implement `metadata`.
6. [x] Implement `possible_agents`.
7. [x] Implement `observation_space(agent)`.
8. [x] Implement `action_space(agent)`.
9. [x] Implement `reset(seed=None, options=None)`.
10. [x] Implement `observe(agent)`.
11. [x] Implement `last()`.
12. [x] Implement `step(action_id)`.
13. [x] Implement termination handling when one player remains.
14. [x] Implement truncation handling when `max_turns` is reached.
15. [x] Add basic `render()` and `close()` support.
16. [x] Add tests using PettingZoo API checks:

```python
from pettingzoo.test import api_test

api_test(env(), num_cycles=1000)
```

## Phase 5: Reward Design

Testable outcome: PettingZoo rollouts produce sensible rewards at game end without changing the underlying engine rules.

1. [x] Start with sparse terminal rewards:
   - [x] winner gets `+1`
   - [x] eliminated players get `-1`
   - [x] unfinished/truncated games get `0`
2. [x] Add tests for reward assignment in:
   - [x] normal win
   - [x] player elimination
   - [x] max-turn truncation
3. [x] Record reward totals in diagnostics through PettingZoo `info["episode_reward"]`.
4. [x] Avoid shaped rewards until baseline training works.
5. [x] Document optional shaped rewards to consider later:
   - [ ] small survival reward
   - [ ] penalty for wasting `defuse`
   - [ ] reward for avoiding a known exploding kitten
   - [ ] reward for causing or increasing opponent risk

## Phase 6: Random PettingZoo Rollouts

Testable outcome: PettingZoo random agents can complete games using the wrapper and masks.

1. [x] Add `scripts/run_pettingzoo_random.py`.
2. [x] Reset the PettingZoo environment with a seed.
3. [x] Iterate with `env.agent_iter()`.
4. [x] Use `env.last()` to get observation, reward, termination, truncation, and info.
5. [x] Sample only legal actions from `action_mask`.
6. [x] Call `env.step(action_id)`.
7. [x] Print a small rollout summary.
8. [x] Add tests or smoke checks that:
   - [x] games finish
   - [x] illegal actions are not sampled
   - [x] reward totals are reasonable
   - [x] results are deterministic for a fixed seed

## Phase 7: Single Learning Agent Against Scripted Opponents

Testable outcome: one trainable agent can learn against fixed scripted opponents.

1. [x] Create a single-agent Gymnasium wrapper around the engine.
2. [x] Choose one learner slot, defaulting to `player_1`.
3. [x] Use scripted opponents for all other players:
   - [x] `RandomAgent`
   - [x] `DrawOnlyAgent`
   - [x] `SafeRuleAgent`
   - [x] `SkipIfPossibleAgent`
   - [x] reserve `HeuristicAgent` as a later optional opponent once that class exists
4. [x] Expose only the learner's observation and action mask.
5. [x] Step scripted opponents automatically until it is the learner's turn again.
6. [x] Add tests that:
   - [x] learner actions apply correctly
   - [x] scripted opponent turns are resolved automatically
   - [x] terminal rewards are assigned correctly
7. [x] Keep the wrapper compatible with single-agent RL libraries such as Stable-Baselines3.
8. [x] Defer actual training scripts, checkpoints, and trained-policy comparisons to Phase 8.

## Phase 8A: WandB Experiment Tracking

1. [x] Add optional `tracking` dependency with `wandb`.
2. [x] Add `scripts/evaluate_single_agent.py`.
3. [x] Log run config: seed, players, opponent strategy, card filters, combo rules.
4. [x] Log aggregate metrics: win rate, reward, turns, truncations.
5. [x] Log per-episode metrics.
6. [x] Log a small WandB Table of selected episode traces.
7. [x] Add a `--wandb-mode` option: `online`, `offline`, or `disabled`.
8. [x] Keep WandB optional so tests and local simulation still run without it.
9. [x] Ignore local experiment outputs such as `wandb/` and `models/`.

## Phase 8B: Mask-Aware PPO Training

Testable outcome: a real RL learner can train against scripted opponents while respecting legal action masks.

1. [x] Add optional `training` dependencies:
   - [x] `stable-baselines3`
   - [x] `sb3-contrib`
2. [x] Confirm whether the current `SingleAgentEnv` observation shape works directly with `MaskablePPO`.
3. [x] Add the small compatibility method needed by `sb3-contrib`: `SingleAgentEnv.action_masks()`.
4. [x] Add `scripts/train_single_agent.py`.
5. [x] Use `MaskablePPO` rather than vanilla PPO so illegal actions are masked during learning.
6. [x] Start with a tiny training configuration:
   - [x] learner: `player_1`
   - [x] opponents: `draw-only`
   - [x] players: `2`
   - [x] short timestep budget
   - [x] fixed seed
7. [x] Save trained models under ignored `models/`.
8. [x] Log training configuration and metrics to WandB:
   - [x] total timesteps
   - [x] opponent strategy
   - [x] seed
   - [x] episode reward
   - [x] episode length
   - [x] final evaluation win rate
9. [x] Add a small smoke test that imports the training helpers without running a long training job.
10. [x] Run one tiny offline WandB training job to prove the plumbing works.
11. [x] Document the training command in `README.md`.

## Phase 8: Broader Training Experiments

Testable outcome: we can run repeatable training jobs and evaluate trained policies against baselines.

1. [x] Add `scripts/evaluate_agent.py` for saved policies.
2. [x] Compare saved policies against:
   - [x] draw-only opponents
   - [x] random opponents
   - [x] safe-rule opponents
3. [x] Log configuration used for each evaluation run.
4. [x] Track evaluation metrics:
   - [x] win rate
   - [x] average turns survived
   - [x] defuses used
   - [x] explosions
   - [x] cards played by type
5. [ ] Run longer training jobs once the tiny masked-PPO job is stable.
6. [ ] Compare learning curves across opponent strategies.
7. [ ] Compare learning curves with and without combo rules enabled.
8. [x] Add README instructions for saved-policy evaluation.
9. [ ] Add README instructions for longer experiments.

## Phase 8C: Visible Experiment Loop

Testable outcome: one command trains a model, evaluates it against baselines, compares it with random masked play, logs the result, and writes a report.

1. [x] Add `scripts/run_training_experiment.py`.
2. [x] Train one MaskablePPO model.
3. [x] Evaluate the trained model against baseline opponents.
4. [x] Evaluate the random masked learner against the same opponents.
5. [x] Print a side-by-side ASCII comparison table.
6. [x] Log comparison metrics to WandB.
7. [x] Log a WandB comparison table.
8. [x] Save an experiment summary JSON under `reports/`.
9. [x] Ignore `reports/` in Git.
10. [x] Add tests for the orchestration/reporting helpers without running long training jobs.
11. [x] Run a tiny end-to-end smoke experiment.

## Phase 8D: Multi-Policy 3-Player Evaluation

Testable outcome: two trained policies and one random baseline can play in the same 3-player game, with seat-by-seat results and combined trained-agent metrics.

1. [x] Train 3-player-compatible single-agent smoke policies.
2. [x] Add `scripts/evaluate_multi_policy_game.py`.
3. [x] Support policy assignment by seat:
   - [x] saved MaskablePPO model via `model:<path>`
   - [x] random policy
   - [x] safe-rule policy
   - [x] draw-only policy
   - [x] skip-if-possible policy
4. [x] Run 3-player games with:
   - [x] `player_1` trained
   - [x] `player_2` trained
   - [x] `player_3` random
5. [x] Track per-seat win rate.
6. [x] Track combined trained-agent win rate.
7. [x] Track random-agent win rate.
8. [x] Track per-seat diagnostics:
   - [x] average reward
   - [x] average turns survived
   - [x] defuses
   - [x] explosions
   - [x] cards played by type
9. [x] Log comparison table to WandB.
10. [x] Add README command examples.
11. [x] Add focused tests without requiring long training runs.
12. [x] Run tiny offline WandB smoke evaluation.

## Phase 9: Multi-Agent Training And Self-Play

Testable outcome: multiple policies can train or evaluate in the same environment.

1. [x] Decide which multi-agent training framework to try first:
   - [x] RLlib
   - [ ] TorchRL
   - [ ] CleanRL multi-agent example
2. [x] Connect the PettingZoo environment to the chosen trainer.
3. [ ] Decide policy sharing strategy:
   - [x] one shared policy for all players
   - [ ] separate policies per player seat
   - [ ] learner policy versus frozen opponent policies
4. [ ] Add self-play evaluation.
5. [x] Save policy checkpoints.
6. [ ] Compare trained policies against scripted baselines.
7. [ ] Track whether seat order creates a strong advantage.

## Phase 9A: RLlib Smoke Training

Testable outcome: RLlib can train a shared policy on the existing PettingZoo environment without selecting illegal masked actions.

1. [x] Add optional `multiagent` dependency with `ray[rllib]`.
2. [x] Add `scripts/train_rllib_pettingzoo.py`.
3. [x] Wrap the existing PettingZoo environment with RLlib's `PettingZooEnv`.
4. [x] Add an RLlib observation wrapper that renames `observation` to `observations` for Ray's action-mask model.
5. [x] Use Ray's old-API `TorchActionMaskModel` so illegal actions are masked during PPO training.
6. [x] Start with a 3-player game and one shared policy for all seats.
7. [x] Train for one tiny iteration as a smoke test.
8. [x] Log RLlib metrics to WandB when requested.
9. [x] Save an RLlib checkpoint under ignored `models/rllib/`.
10. [x] Add smoke tests for the wrapper/config helpers without running full training.
11. [x] Document the RLlib smoke command in `README.md`.

Notes:

- This first RLlib path intentionally uses RLlib's old API stack because Ray's bundled action-mask model is currently old-stack based.
- The smoke run is only a plumbing test. It proves masked shared-policy training can run; it does not prove the policy has learned a good strategy yet.

## Phase 9B: Roadmap To Serious Self-Play

Testable outcome: we have a clear, staged path from RLlib smoke training to a repeatable self-play system with checkpoint evaluation, baseline comparisons, WandB visibility, and policy-pool training.

### 9B.1 RLlib Checkpoint Evaluation

Testable outcome: saved RLlib checkpoints can be loaded and evaluated against scripted and learned opponents.

1. [ ] Add `scripts/evaluate_rllib_checkpoint.py`.
2. [ ] Load an RLlib checkpoint saved by `scripts/train_rllib_pettingzoo.py`.
3. [ ] Support one shared RLlib policy for all learner-controlled seats.
4. [ ] Support scripted opponent seats:
   - [ ] `random`
   - [ ] `safe-rule`
   - [ ] `draw-only`
   - [ ] `skip-if-possible`
5. [ ] Support seat assignment syntax similar to multi-policy evaluation:
   - [ ] `player_1=rllib:<checkpoint_path>`
   - [ ] `player_2=random`
   - [ ] `player_3=safe-rule`
6. [ ] Evaluate fixed checkpoints over many games with deterministic seeds.
7. [ ] Report per-seat metrics:
   - [ ] win rate
   - [ ] average reward
   - [ ] average turns survived
   - [ ] explosions
   - [ ] defuses
   - [ ] cards played by type
8. [ ] Report grouped metrics:
   - [ ] RLlib-controlled win rate
   - [ ] scripted-opponent win rate
   - [ ] seat-order advantage
9. [ ] Log the evaluation table to WandB.
10. [ ] Add focused tests for checkpoint-evaluation helpers without requiring a long RLlib run.
11. [ ] Run one tiny RLlib checkpoint evaluation smoke test.

### 9B.2 Training Configuration Files

Testable outcome: longer RLlib jobs can be started from versionable config files instead of fragile command lines.

1. [ ] Add an RLlib training config YAML format.
2. [ ] Include core game settings:
   - [ ] players
   - [ ] max turns
   - [ ] seed
   - [ ] included cards
   - [ ] excluded cards
   - [ ] enabled combo rules
   - [ ] hidden-information flags
3. [ ] Include PPO settings:
   - [ ] iterations
   - [ ] learning rate
   - [ ] train batch size
   - [ ] minibatch size
   - [ ] rollout fragment length
   - [ ] number of epochs
   - [ ] number of environment runners
4. [ ] Include experiment settings:
   - [ ] run name
   - [ ] checkpoint interval
   - [ ] evaluation interval
   - [ ] WandB mode
   - [ ] WandB project
5. [ ] Save the resolved config beside each checkpoint.
6. [ ] Add README examples for smoke, medium, and longer runs.

### 9B.3 Serious Shared-Policy Training

Testable outcome: one shared RLlib policy trains for long enough to produce meaningful baseline comparisons.

1. [ ] Add checkpoint intervals during training.
2. [ ] Add evaluation intervals during training.
3. [ ] Evaluate against scripted baselines during training:
   - [ ] random
   - [ ] safe-rule
   - [ ] draw-only
4. [ ] Log evaluation metrics to WandB as separate charts from training metrics.
5. [ ] Track action distribution during evaluation.
6. [ ] Track illegal action rate, expected to stay at `0`.
7. [ ] Track cards played by type.
8. [ ] Track combo use.
9. [ ] Run a medium shared-policy training job.
10. [ ] Compare the final checkpoint against the first checkpoint and scripted baselines.

### 9B.4 Self-Play Policy Pool

Testable outcome: training can sample opponents from a small pool of scripted policies and previous checkpoints.

1. [ ] Define a policy-pool data structure.
2. [ ] Add pool entries for scripted policies:
   - [ ] random
   - [ ] safe-rule
   - [ ] draw-only
3. [ ] Add pool entries for RLlib checkpoints:
   - [ ] latest checkpoint
   - [ ] previous checkpoint
   - [ ] best checkpoint by evaluation win rate
4. [ ] Add opponent sampling modes:
   - [ ] latest only
   - [ ] random historical checkpoint
   - [ ] mix scripted and historical opponents
5. [ ] Save policy-pool metadata under `reports/` or beside checkpoints.
6. [ ] Add tests for opponent sampling without running training.
7. [ ] Run a tiny policy-pool smoke test.

### 9B.5 Frozen Opponent And League Training

Testable outcome: a learner can train against frozen historical policies without overwriting the opponent policy during the same update loop.

1. [ ] Decide whether RLlib should use:
   - [ ] one trainable shared policy only
   - [ ] one trainable learner policy plus frozen opponent policies
   - [ ] separate trainable policies per seat
2. [ ] Add policy mapping for learner-versus-frozen-opponent training.
3. [ ] Load frozen opponent checkpoints into RLlib policies.
4. [ ] Ensure frozen policies do not receive optimizer updates.
5. [ ] Evaluate current learner against the frozen pool.
6. [ ] Promote checkpoints into the pool when they beat baselines.
7. [ ] Add tests for policy mapping and frozen-policy configuration.

### 9B.6 Seat-Bias And Robustness Evaluation

Testable outcome: we can tell whether an apparent win rate is real skill or mostly seat order.

1. [ ] Rotate policy assignment across seats during evaluation.
2. [ ] Report win rate by seat.
3. [ ] Report win rate by policy independent of seat.
4. [ ] Run evaluation with several seeds.
5. [ ] Add confidence intervals or standard error for win rate.
6. [ ] Track average game length by seat.
7. [ ] Add a WandB table for seat-rotation results.

### 9B.7 Reward And Observation Refinement

Testable outcome: reward and observation changes can be tested without breaking the engine's hidden-information boundary.

1. [ ] Keep sparse rewards as the default.
2. [ ] Add optional reward-shaping config:
   - [ ] small survival reward
   - [ ] penalty for elimination
   - [ ] reward for forcing an opponent elimination
   - [ ] penalty for wasting `defuse`
3. [ ] Add tests proving shaped rewards do not change game rules.
4. [ ] Add observation ablation flags:
   - [ ] include known top cards
   - [ ] include opponent hand sizes
   - [ ] include public event counts
5. [ ] Compare learning curves with and without shaping.
6. [ ] Compare learning curves with and without richer observations.

### 9B.8 Definition Of Serious Self-Play

We can call the setup serious when all of these are true:

1. [ ] Training can run from a config file for hundreds or thousands of RLlib iterations.
2. [ ] Checkpoints are saved and can be restored reliably.
3. [ ] Evaluation can compare RLlib checkpoints against scripted baselines and historical checkpoints.
4. [ ] WandB shows training curves, evaluation win rates, seat-bias metrics, and action/card diagnostics.
5. [ ] Opponents can be sampled from a policy pool.
6. [ ] Frozen opponent policies can be used during training.
7. [ ] Seat rotation is part of evaluation.
8. [ ] The best policy beats random and draw-only baselines consistently.
9. [ ] The best policy is competitive with or better than `safe-rule`.
10. [ ] No agent observes hidden draw-pile order or exact opponent hands.

## Phase 10: Search And Imperfect-Information Agents

Testable outcome: agents can reason about hidden information without cheating.

1. [ ] Add a belief-state helper that estimates unknown cards.
2. [ ] Add a rollout agent that samples possible hidden states.
3. [ ] Add a Monte Carlo evaluation agent.
4. [ ] Consider Information Set MCTS.
5. [ ] Consider OpenSpiel only if we want deeper game-theory algorithms.
6. [ ] Add tests that ensure search agents do not inspect:
   - [ ] actual draw pile order
   - [ ] exact opponent hands
   - [ ] hidden random steal results before they happen

## Recommended Immediate Next Tasks

1. [x] Implement `SafeRuleAgent`.
2. [x] Add `ActionEncoder` and action mask tests.
3. [x] Add `ObservationEncoder` and hidden-information tests.
4. [x] Add the PettingZoo AEC wrapper.
5. [x] Run PettingZoo `api_test`.
