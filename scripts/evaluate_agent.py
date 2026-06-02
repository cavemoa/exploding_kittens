from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from exploding_kittens import SingleAgentEnv


WANDB_MODES = ("online", "offline", "disabled")
DEFAULT_OPPONENT_STRATEGIES = ("draw-only", "random", "safe-rule")
COUNTED_CARD_EVENTS = (
    "skip",
    "attack",
    "shuffle",
    "favor",
    "alter_the_future",
    "see_the_future",
)


@dataclass(frozen=True)
class PolicyEpisodeResult:
    episode: int
    seed: int
    opponent_strategy: str
    reward: float
    win: bool
    turns_survived: int
    truncated: bool
    winner: str | None
    learner_actions: int
    defuses: int
    explosions: int
    cards_played: dict[str, int]


@dataclass(frozen=True)
class OpponentEvaluationResult:
    opponent_strategy: str
    episodes: int
    wins: int
    win_rate: float
    average_reward: float
    average_turns_survived: float
    truncations: int
    truncation_rate: float
    defuses: int
    explosions: int
    cards_played: dict[str, int]
    episode_results: tuple[PolicyEpisodeResult, ...]


@dataclass(frozen=True)
class SavedPolicyEvaluationResult:
    model_path: Path
    learner: str
    players: int
    opponent_results: tuple[OpponentEvaluationResult, ...]


def evaluate_saved_policy(
    *,
    model_path: Path,
    episodes: int = 10,
    players: int = 2,
    learner: str = "player_1",
    opponent_strategies: tuple[str, ...] = DEFAULT_OPPONENT_STRATEGIES,
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    deterministic: bool = True,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> SavedPolicyEvaluationResult:
    validate_evaluation_args(
        model_path=model_path,
        episodes=episodes,
        opponent_strategies=opponent_strategies,
        wandb_mode=wandb_mode,
    )
    config = {
        "model_path": str(model_path),
        "episodes": episodes,
        "players": players,
        "learner": learner,
        "opponent_strategies": opponent_strategies,
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "deterministic": deterministic,
        "policy_type": "saved_maskable_ppo",
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )

    model = load_maskable_ppo(model_path)
    try:
        opponent_results = tuple(
            evaluate_policy_against_opponent(
                model,
                episodes=episodes,
                players=players,
                learner=learner,
                opponent_strategy=opponent_strategy,
                seed=seed + strategy_index * 10_000,
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                deterministic=deterministic,
            )
            for strategy_index, opponent_strategy in enumerate(opponent_strategies)
        )
        result = SavedPolicyEvaluationResult(
            model_path=model_path,
            learner=learner,
            players=players,
            opponent_results=opponent_results,
        )
        log_results(run, result)
        return result
    finally:
        if run is not None:
            run.finish()


def evaluate_policy_against_opponent(
    model: Any,
    *,
    episodes: int,
    players: int,
    learner: str,
    opponent_strategy: str,
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    deterministic: bool,
) -> OpponentEvaluationResult:
    episode_results = [
        run_episode(
            model,
            episode=episode_index,
            players=players,
            learner=learner,
            opponent_strategy=opponent_strategy,
            seed=seed + episode_index,
            max_turns=max_turns,
            include_cards=include_cards,
            exclude_cards=exclude_cards,
            enabled_combo_rules=enabled_combo_rules,
            reveal_opponent_card_counts=reveal_opponent_card_counts,
            deterministic=deterministic,
        )
        for episode_index in range(episodes)
    ]
    return summarize_opponent_results(opponent_strategy, episode_results)


def run_episode(
    model: Any,
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
    deterministic: bool,
) -> PolicyEpisodeResult:
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
    processed_events = len(environment.engine.events)
    event_counts: Counter[str] = Counter()
    cards_played: Counter[str] = Counter()
    info: dict[str, Any] = {"winner": None, "turn_count": 0}

    while not terminated and not truncated:
        action, _ = model.predict(
            observation,
            deterministic=deterministic,
            action_masks=environment.action_masks(),
        )
        observation, reward, terminated, truncated, info = environment.step(int(action))
        total_reward += reward
        learner_actions += 1

        new_events = environment.engine.events[processed_events:]
        processed_events = len(environment.engine.events)
        for event in new_events:
            event_counts[event.kind] += 1
            if event.kind in COUNTED_CARD_EVENTS:
                cards_played[event.kind] += 1

    environment.close()
    return PolicyEpisodeResult(
        episode=episode,
        seed=seed,
        opponent_strategy=opponent_strategy,
        reward=total_reward,
        win=info["winner"] == learner,
        turns_survived=info["turn_count"],
        truncated=truncated,
        winner=info["winner"],
        learner_actions=learner_actions,
        defuses=event_counts["defuse"],
        explosions=event_counts["eliminated"],
        cards_played=dict(sorted(cards_played.items())),
    )


def summarize_opponent_results(
    opponent_strategy: str,
    episode_results: list[PolicyEpisodeResult],
) -> OpponentEvaluationResult:
    episodes = len(episode_results)
    wins = sum(result.win for result in episode_results)
    rewards = [result.reward for result in episode_results]
    turns_survived = [result.turns_survived for result in episode_results]
    truncations = sum(result.truncated for result in episode_results)
    cards_played: Counter[str] = Counter()
    for result in episode_results:
        cards_played.update(result.cards_played)

    return OpponentEvaluationResult(
        opponent_strategy=opponent_strategy,
        episodes=episodes,
        wins=wins,
        win_rate=wins / episodes,
        average_reward=float(np.mean(rewards)),
        average_turns_survived=float(np.mean(turns_survived)),
        truncations=truncations,
        truncation_rate=truncations / episodes,
        defuses=sum(result.defuses for result in episode_results),
        explosions=sum(result.explosions for result in episode_results),
        cards_played=dict(sorted(cards_played.items())),
        episode_results=tuple(episode_results),
    )


def validate_evaluation_args(
    *,
    model_path: Path,
    episodes: int,
    opponent_strategies: tuple[str, ...],
    wandb_mode: str,
) -> None:
    if not model_path.exists():
        raise ValueError(f"Model file does not exist: {model_path}")
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if not opponent_strategies:
        raise ValueError("At least one opponent strategy is required.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


def load_maskable_ppo(model_path: Path) -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            'Saved-policy evaluation requires `pip install -e ".[training]"` first.'
        ) from exc
    return MaskablePPO.load(model_path)


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
        job_type="saved-policy-evaluation",
    )


def log_results(run: Any | None, result: SavedPolicyEvaluationResult) -> None:
    if run is None:
        return

    for opponent_result in result.opponent_results:
        prefix = f"eval/{opponent_result.opponent_strategy}"
        run.log(
            {
                f"{prefix}/win_rate": opponent_result.win_rate,
                f"{prefix}/average_reward": opponent_result.average_reward,
                f"{prefix}/average_turns_survived": opponent_result.average_turns_survived,
                f"{prefix}/truncation_rate": opponent_result.truncation_rate,
                f"{prefix}/defuses": opponent_result.defuses,
                f"{prefix}/explosions": opponent_result.explosions,
            }
        )
    log_comparison_table(run, result)


def log_comparison_table(run: Any | None, result: SavedPolicyEvaluationResult) -> None:
    if run is None:
        return

    import wandb

    table = wandb.Table(
        columns=[
            "opponent_strategy",
            "episodes",
            "wins",
            "win_rate",
            "average_reward",
            "average_turns_survived",
            "truncation_rate",
            "defuses",
            "explosions",
            "cards_played",
        ]
    )
    for opponent_result in result.opponent_results:
        table.add_data(
            opponent_result.opponent_strategy,
            opponent_result.episodes,
            opponent_result.wins,
            opponent_result.win_rate,
            opponent_result.average_reward,
            opponent_result.average_turns_survived,
            opponent_result.truncation_rate,
            opponent_result.defuses,
            opponent_result.explosions,
            opponent_result.cards_played,
        )
    run.log({"eval/opponent_comparison": table})


def format_results_table(result: SavedPolicyEvaluationResult) -> str:
    headers = (
        "opponent",
        "episodes",
        "wins",
        "win_rate",
        "avg_reward",
        "avg_turns",
        "trunc_rate",
    )
    rows = [
        (
            opponent_result.opponent_strategy,
            str(opponent_result.episodes),
            str(opponent_result.wins),
            f"{opponent_result.win_rate:.3f}",
            f"{opponent_result.average_reward:.3f}",
            f"{opponent_result.average_turns_survived:.1f}",
            f"{opponent_result.truncation_rate:.3f}",
        )
        for opponent_result in result.opponent_results
    ]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in rows))
        for column in range(len(headers))
    ]
    separator = "+".join("-" * (width + 2) for width in widths)
    lines = [
        separator,
        "|".join(
            f" {headers[column]:<{widths[column]}} "
            for column in range(len(headers))
        ),
        separator,
    ]
    for row in rows:
        lines.append(
            "|".join(
                f" {row[column]:<{widths[column]}} "
                for column in range(len(headers))
            )
        )
    lines.append(separator)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MaskablePPO model against scripted opponents."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--players", type=int, default=2)
    parser.add_argument("--learner", default="player_1")
    parser.add_argument(
        "--opponent-strategies",
        type=parse_name_list,
        default=DEFAULT_OPPONENT_STRATEGIES,
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
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
    result = evaluate_saved_policy(
        model_path=args.model_path,
        episodes=args.episodes,
        players=args.players,
        learner=args.learner,
        opponent_strategies=args.opponent_strategies,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        deterministic=not args.stochastic,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("Saved Policy Evaluation")
    print(f"Model: {result.model_path}")
    print(f"Learner: {result.learner}")
    print(f"Players: {result.players}")
    print(format_results_table(result))


if __name__ == "__main__":
    main()
