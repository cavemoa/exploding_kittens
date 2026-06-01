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
4. [ ] Add a `SafeRuleAgent` that proves the observation shape is usable.
5. [ ] Add tests showing `SafeRuleAgent` can:
   - [ ] avoid drawing when `known_top_cards[0] == "exploding_kitten"`
   - [ ] prefer `shuffle`, `skip`, `attack`, or `alter_the_future` when drawing is dangerous
   - [ ] avoid spending `defuse` in `TwoOfAKindRule`
   - [ ] prefer low-value pairs such as `normal`, `normal`

## Phase 2: Define A Stable Integer Action Space

Testable outcome: every legal `GameAction` can be represented by a stable integer action ID, and illegal action IDs can be masked out.

1. [ ] Create `src/exploding_kittens/action_encoding.py`.
2. [ ] Define an `ActionEncoder` class.
3. [ ] Define a stable action ID layout, for example:
   - [ ] draw
   - [ ] play one card from hand slot
   - [ ] play targeted card from hand slot against target player
   - [ ] play two-card combo from hand slots against target player
   - [ ] reserve space for future choices such as named-card requests
4. [ ] Decide maximum supported hand size for the first RL wrapper.
5. [ ] Decide maximum supported player count for the first RL wrapper.
6. [ ] Implement `encode(action: GameAction, observation: GameObservation) -> int`.
7. [ ] Implement `decode(action_id: int, observation: GameObservation) -> GameAction`.
8. [ ] Implement `action_mask(observation: GameObservation) -> np.ndarray`.
9. [ ] Add tests that:
   - [ ] every current legal action maps to exactly one valid action ID
   - [ ] every valid action ID decodes back to the expected `GameAction`
   - [ ] illegal action IDs are masked out
   - [ ] target-specific actions preserve the selected target player
   - [ ] combo actions preserve the selected card indexes

## Phase 3: Define A Numeric Observation Encoder

Testable outcome: `GameObservation` can be converted into fixed-shape numeric arrays suitable for Gymnasium, PettingZoo, and RL libraries.

1. [ ] Create `src/exploding_kittens/observation_encoding.py`.
2. [ ] Define an `ObservationEncoder` class.
3. [ ] Encode own hand by fixed hand slots.
4. [ ] Encode own hand counts by card type.
5. [ ] Encode discard pile counts by card type.
6. [ ] Encode draw pile size.
7. [ ] Encode living player count.
8. [ ] Encode current attack turn debt.
9. [ ] Encode next-player attack turn debt.
10. [ ] Encode `known_top_cards` from `SeeTheFutureCard`.
11. [ ] Encode opponent hand sizes only when configured.
12. [ ] Return an observation dictionary like:

```python
{
    "observation": np.ndarray,
    "action_mask": np.ndarray,
}
```

13. [ ] Add tests that:
   - [ ] encoded observations have stable shapes
   - [ ] encoded observations do not expose draw pile order
   - [ ] encoded observations do not expose exact opponent hands
   - [ ] `known_top_cards` appears only after `SeeTheFutureCard`
   - [ ] action masks match `engine.legal_actions()`

## Phase 4: Add The PettingZoo AEC Environment

Testable outcome: the simulator can run as a valid PettingZoo AEC environment.

1. [ ] Add PettingZoo and Gymnasium to project dependencies.
2. [ ] Create `src/exploding_kittens/pettingzoo_env.py`.
3. [ ] Add an `env()` factory function.
4. [ ] Add a `raw_env` class inheriting from `pettingzoo.AECEnv`.
5. [ ] Implement `metadata`.
6. [ ] Implement `possible_agents`.
7. [ ] Implement `observation_space(agent)`.
8. [ ] Implement `action_space(agent)`.
9. [ ] Implement `reset(seed=None, options=None)`.
10. [ ] Implement `observe(agent)`.
11. [ ] Implement `last()`.
12. [ ] Implement `step(action_id)`.
13. [ ] Implement termination handling when one player remains.
14. [ ] Implement truncation handling when `max_turns` is reached.
15. [ ] Add `render()` later only if needed.
16. [ ] Add tests using PettingZoo API checks:

```python
from pettingzoo.test import api_test

api_test(env(), num_cycles=1000)
```

## Phase 5: Reward Design

Testable outcome: PettingZoo rollouts produce sensible rewards at game end without changing the underlying engine rules.

1. [ ] Start with sparse terminal rewards:
   - [ ] winner gets `+1`
   - [ ] eliminated players get `-1`
   - [ ] unfinished/truncated games get `0`
2. [ ] Add tests for reward assignment in:
   - [ ] normal win
   - [ ] player elimination
   - [ ] max-turn truncation
3. [ ] Record reward totals in diagnostics.
4. [ ] Avoid shaped rewards until baseline training works.
5. [ ] Later consider optional shaped rewards:
   - [ ] small survival reward
   - [ ] penalty for wasting `defuse`
   - [ ] reward for avoiding a known exploding kitten
   - [ ] reward for causing or increasing opponent risk

## Phase 6: Random PettingZoo Rollouts

Testable outcome: PettingZoo random agents can complete games using the wrapper and masks.

1. [ ] Add `scripts/run_pettingzoo_random.py`.
2. [ ] Reset the PettingZoo environment with a seed.
3. [ ] Iterate with `env.agent_iter()`.
4. [ ] Use `env.last()` to get observation, reward, termination, truncation, and info.
5. [ ] Sample only legal actions from `action_mask`.
6. [ ] Call `env.step(action_id)`.
7. [ ] Print a small rollout summary.
8. [ ] Add tests or smoke checks that:
   - [ ] games finish
   - [ ] illegal actions are not sampled
   - [ ] reward totals are reasonable
   - [ ] results are deterministic for a fixed seed

## Phase 7: Single Learning Agent Against Scripted Opponents

Testable outcome: one trainable agent can learn against fixed scripted opponents.

1. [ ] Create a single-agent Gymnasium wrapper around the engine.
2. [ ] Choose one learner slot, for example `player_1`.
3. [ ] Use scripted opponents for all other players:
   - [ ] `RandomAgent`
   - [ ] `SafeRuleAgent`
   - [ ] later `HeuristicAgent`
4. [ ] Expose only the learner's observation and action mask.
5. [ ] Step scripted opponents automatically until it is the learner's turn again.
6. [ ] Add tests that:
   - [ ] learner actions apply correctly
   - [ ] scripted opponent turns are resolved automatically
   - [ ] terminal rewards are assigned correctly
7. [ ] Train first with Stable-Baselines3 or another single-agent RL library.
8. [ ] Compare the trained learner against:
   - [ ] random opponents
   - [ ] draw-only opponents
   - [ ] safe-rule opponents

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

1. [ ] Implement `SafeRuleAgent`.
2. [ ] Add `ActionEncoder` and action mask tests.
3. [ ] Add `ObservationEncoder` and hidden-information tests.
4. [ ] Add the PettingZoo AEC wrapper.
5. [ ] Run PettingZoo `api_test`.

