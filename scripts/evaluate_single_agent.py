from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from exploding_kittens import SingleAgentEnv


WANDB_MODES = ("online", "offline", "disabled")
TRACE_COLUMNS = (
    "episode",
    "step",
    "turn_count",
    "action_id",
    "action_kind",
    "card_indexes",
    "target_player",
    "reward",
    "winner",
    "draw_pile_size",
    "living_players",
    "known_top_cards",
    "legal_action_count",
)


@dataclass(frozen=True)
class EpisodeResult:
    episode: int
    seed: int
    reward: float
    win: bool
    turns: int
    truncated: bool
    winner: str | None
    learner_actions: int


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    wins: int
    win_rate: float
    average_reward: float
    average_turns: float
    truncations: int
    truncation_rate: float
    episode_results: tuple[EpisodeResult, ...]
    trace_rows: tuple[tuple[Any, ...], ...]


def evaluate_single_agent(
    *,
    episodes: int = 10,
    players: int = 3,
    learner: str = "player_1",
    opponent_strategy: str = "safe-rule",
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    trace_episodes: int = 3,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> EvaluationResult:
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if trace_episodes < 0:
        raise ValueError("trace_episodes cannot be negative.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")

    config = {
        "episodes": episodes,
        "players": players,
        "learner": learner,
        "opponent_strategy": opponent_strategy,
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "trace_episodes": trace_episodes,
        "learner_policy": "random_masked",
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    rng = np.random.default_rng(seed)
    episode_results: list[EpisodeResult] = []
    trace_rows: list[tuple[Any, ...]] = []
    try:
        for episode_index in range(episodes):
            episode_seed = seed + episode_index
            episode_result, episode_trace = run_episode(
                episode=episode_index,
                players=players,
                learner=learner,
                opponent_strategy=opponent_strategy,
                seed=episode_seed,
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                rng=rng,
                collect_trace=episode_index < trace_episodes,
            )
            episode_results.append(episode_result)
            trace_rows.extend(episode_trace)
            log_episode(run, episode_result)

        result = summarize_results(episode_results, trace_rows)
        log_summary(run, result)
        log_trace_table(run, result.trace_rows)
        return result
    finally:
        if run is not None:
            run.finish()


def run_episode(
    *,
    episode: int,
    players: int,
    learner: str,
    opponent_strategy: str,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    rng: np.random.Generator,
    collect_trace: bool,
) -> tuple[EpisodeResult, list[tuple[Any, ...]]]:
    environment = SingleAgentEnv(
        players=players,
        learner=learner,
        opponent_strategy=opponent_strategy,
        seed=seed,
        max_turns=max_turns,
        include_cards=include_cards,
        exclude_cards=exclude_cards,
        enabled_combo_rules=enabled_combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )
    observation, _ = environment.reset(seed=seed)
    terminated = False
    truncated = False
    total_reward = 0.0
    learner_actions = 0
    trace_rows: list[tuple[Any, ...]] = []

    while not terminated and not truncated:
        engine_observation = environment.engine.observe(learner)
        action_id = sample_masked_action(observation["action_mask"], rng)
        game_action = environment.action_encoder.decode(action_id, engine_observation)
        observation, reward, terminated, truncated, info = environment.step(action_id)
        total_reward += reward
        learner_actions += 1

        if collect_trace:
            trace_rows.append(
                (
                    episode,
                    learner_actions,
                    info["turn_count"],
                    action_id,
                    game_action.kind.value,
                    ",".join(str(index) for index in game_action.card_indexes),
                    game_action.target_player or "",
                    reward,
                    info["winner"] or "",
                    engine_observation.draw_pile_size,
                    engine_observation.living_players,
                    ",".join(engine_observation.known_top_cards),
                    int(observation["action_mask"].sum()),
                )
            )

    winner = info["winner"]
    environment.close()
    return (
        EpisodeResult(
            episode=episode,
            seed=seed,
            reward=total_reward,
            win=winner == learner,
            turns=info["turn_count"],
            truncated=truncated,
            winner=winner,
            learner_actions=learner_actions,
        ),
        trace_rows,
    )


def summarize_results(
    episode_results: list[EpisodeResult],
    trace_rows: list[tuple[Any, ...]],
) -> EvaluationResult:
    episodes = len(episode_results)
    wins = sum(result.win for result in episode_results)
    truncations = sum(result.truncated for result in episode_results)
    rewards = [result.reward for result in episode_results]
    turns = [result.turns for result in episode_results]
    return EvaluationResult(
        episodes=episodes,
        wins=wins,
        win_rate=wins / episodes,
        average_reward=float(np.mean(rewards)),
        average_turns=float(np.mean(turns)),
        truncations=truncations,
        truncation_rate=truncations / episodes,
        episode_results=tuple(episode_results),
        trace_rows=tuple(trace_rows),
    )


def sample_masked_action(action_mask: np.ndarray, rng: np.random.Generator) -> int:
    legal_actions = np.flatnonzero(action_mask)
    if len(legal_actions) == 0:
        raise ValueError("Cannot sample an action from an empty action mask.")
    return int(rng.choice(legal_actions))


def start_wandb_run(
    *,
    mode: str,
    project: str,
    entity: str | None,
    config: dict[str, Any],
) -> Any | None:
    if mode == "disabled":
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'WandB tracking requires `pip install -e ".[tracking]"` first.'
        ) from exc

    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        config=config,
        job_type="evaluation",
    )


def log_episode(run: Any | None, result: EpisodeResult) -> None:
    if run is None:
        return
    run.log(
        {
            "episode/reward": result.reward,
            "episode/win": int(result.win),
            "episode/turns": result.turns,
            "episode/truncated": int(result.truncated),
            "episode/learner_actions": result.learner_actions,
        },
        step=result.episode,
    )


def log_summary(run: Any | None, result: EvaluationResult) -> None:
    if run is None:
        return
    run.log(
        {
            "eval/win_rate": result.win_rate,
            "eval/average_reward": result.average_reward,
            "eval/average_turns": result.average_turns,
            "eval/truncation_rate": result.truncation_rate,
            "eval/episodes": result.episodes,
        }
    )


def log_trace_table(run: Any | None, trace_rows: tuple[tuple[Any, ...], ...]) -> None:
    if run is None or not trace_rows:
        return
    import wandb

    table = wandb.Table(columns=list(TRACE_COLUMNS))
    for row in trace_rows:
        table.add_data(*row)
    run.log({"traces/episodes": table})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a random masked learner in the single-agent wrapper."
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--learner", default="player_1")
    parser.add_argument("--opponent-strategy", default="safe-rule")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--trace-episodes", type=int, default=3)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    parser.add_argument("--wandb-project", default="exploding-kittens")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def parse_optional_name_list(value: str) -> tuple[str, ...] | None:
    names = parse_name_list(value)
    return names or None


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def main() -> None:
    args = parse_args()
    result = evaluate_single_agent(
        episodes=args.episodes,
        players=args.players,
        learner=args.learner,
        opponent_strategy=args.opponent_strategy,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        trace_episodes=args.trace_episodes,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("Single-Agent Evaluation")
    print(f"Episodes: {result.episodes}")
    print(f"Wins: {result.wins}")
    print(f"Win rate: {result.win_rate:.3f}")
    print(f"Average reward: {result.average_reward:.3f}")
    print(f"Average turns: {result.average_turns:.1f}")
    print(f"Truncation rate: {result.truncation_rate:.3f}")
    print(f"Trace rows: {len(result.trace_rows)}")


if __name__ == "__main__":
    main()
