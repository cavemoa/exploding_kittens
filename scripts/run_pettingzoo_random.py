from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from exploding_kittens import pettingzoo_env


@dataclass(frozen=True)
class PettingZooRolloutResult:
    turns: int
    agent_steps: int
    winner: str | None
    completed: bool
    truncated: bool
    illegal_actions_sampled: int
    episode_rewards: dict[str, float]


def run_random_rollout(
    *,
    players: int = 2,
    seed: int = 1,
    max_turns: int = 500,
    enabled_combo_rules: tuple[str, ...] = (),
) -> PettingZooRolloutResult:
    environment = pettingzoo_env(
        players=players,
        seed=seed,
        max_turns=max_turns,
        enabled_combo_rules=enabled_combo_rules,
    )
    rng = np.random.default_rng(seed)
    rewards = {f"player_{index + 1}": 0.0 for index in range(players)}
    latest_winner: str | None = None
    completed = False
    truncated = False
    illegal_actions_sampled = 0
    agent_steps = 0

    environment.reset(seed=seed)
    for agent in environment.agent_iter():
        observation, reward, termination, truncation, info = environment.last()
        rewards[agent] += float(reward)
        latest_winner = info.get("winner") or latest_winner

        if termination or truncation:
            completed = completed or termination
            truncated = truncated or truncation
            action = None
        else:
            action = sample_masked_action(observation["action_mask"], rng)
            if observation["action_mask"][action] != 1:
                illegal_actions_sampled += 1

        environment.step(action)
        agent_steps += 1

    raw_environment = environment.unwrapped
    turns = raw_environment.engine.turn_count if raw_environment.engine is not None else 0
    environment.close()
    return PettingZooRolloutResult(
        turns=turns,
        agent_steps=agent_steps,
        winner=latest_winner,
        completed=completed,
        truncated=truncated,
        illegal_actions_sampled=illegal_actions_sampled,
        episode_rewards=rewards,
    )


def sample_masked_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        raise ValueError("Cannot sample an action from an empty action mask.")
    return int(rng.choice(legal_actions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one random PettingZoo rollout.")
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    return parser.parse_args()


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main() -> None:
    args = parse_args()
    result = run_random_rollout(
        players=args.players,
        seed=args.seed,
        max_turns=args.max_turns,
        enabled_combo_rules=args.enabled_combo_rules,
    )

    print("PettingZoo Random Rollout")
    print(f"Players: {args.players}")
    print(f"Seed: {args.seed}")
    print(f"Max turns: {args.max_turns}")
    print(f"Turns: {result.turns}")
    print(f"Agent steps: {result.agent_steps}")
    print(f"Winner: {result.winner or 'none'}")
    print(f"Completed: {result.completed}")
    print(f"Truncated: {result.truncated}")
    print(f"Illegal actions sampled: {result.illegal_actions_sampled}")
    print("Episode rewards:")
    for player, reward in sorted(result.episode_rewards.items()):
        print(f"  {player}: {reward:.2f}")


if __name__ == "__main__":
    main()
