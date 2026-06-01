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

## Phase 8: First Training Experiments

Testable outcome: we can run repeatable training jobs and evaluate trained policies against baselines.

1. [ ] Add `scripts/train_single_agent.py`.
2. [ ] Add `scripts/evaluate_agent.py`.
3. [ ] Save trained models under an ignored directory such as `models/`.
4. [ ] Log configuration used for each run.
5. [ ] Track evaluation metrics:
   - [ ] win rate
   - [ ] average turns survived
   - [ ] defuses used
   - [ ] explosions
   - [ ] cards played by type
6. [ ] Run a tiny training job to prove the plumbing works.
7. [ ] Run a longer training job once the tiny job is stable.
8. [ ] Add README instructions for training and evaluation.

## Phase 9: Multi-Agent Training And Self-Play

Testable outcome: multiple policies can train or evaluate in the same environment.

1. [ ] Decide which multi-agent training framework to try first:
   - [ ] RLlib
   - [ ] TorchRL
   - [ ] CleanRL multi-agent example
2. [ ] Connect the PettingZoo environment to the chosen trainer.
3. [ ] Decide policy sharing strategy:
   - [ ] one shared policy for all players
   - [ ] separate policies per player seat
   - [ ] learner policy versus frozen opponent policies
4. [ ] Add self-play evaluation.
5. [ ] Save policy checkpoints.
6. [ ] Compare trained policies against scripted baselines.
7. [ ] Track whether seat order creates a strong advantage.

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
