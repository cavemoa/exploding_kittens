from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Protocol
import random

import numpy as np

from exploding_kittens import (
    ActionEncoder,
    DrawOnlyAgent,
    GameAction,
    GameEngine,
    GameObservation,
    ObservationEncoder,
    RandomAgent,
    SafeRuleAgent,
    SkipIfPossibleAgent,
)


WANDB_MODES = ("online", "offline", "disabled")
SCRIPTED_POLICY_KINDS = ("draw-only", "random", "safe-rule", "skip-if-possible")
COUNTED_CARD_EVENTS = (
    "skip",
    "attack",
    "shuffle",
    "favor",
    "alter_the_future",
    "see_the_future",
    "two_of_a_kind",
)


class SeatController(Protocol):
    policy_label: str
    is_trained: bool

    def choose_action(
        self,
        observation: GameObservation,
        encoded_observation: dict[str, np.ndarray],
        action_encoder: ActionEncoder,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ) -> GameAction:
        """Choose one concrete action for this seat."""


@dataclass(frozen=True)
class SeatPolicySpec:
    seat: str
    policy: str

    @property
    def is_model_policy(self) -> bool:
        return self.policy.startswith("model:")

    @property
    def model_path(self) -> Path | None:
        if not self.is_model_policy:
            return None
        return Path(self.policy.removeprefix("model:"))

    @property
    def label(self) -> str:
        if self.is_model_policy:
            model_path = self.model_path
            return f"model:{model_path.name if model_path is not None else ''}"
        return self.policy


@dataclass(frozen=True)
class MultiPolicyEpisodeResult:
    episode: int
    seed: int
    winner: str | None
    truncated: bool
    turns: int
    rewards: dict[str, float]
    turns_survived: dict[str, int]
    defuses: dict[str, int]
    explosions: dict[str, int]
    cards_played: dict[str, dict[str, int]]


@dataclass(frozen=True)
class SeatEvaluationResult:
    seat: str
    policy_label: str
    wins: int
    win_rate: float
    average_reward: float
    average_turns_survived: float
    defuses: int
    explosions: int
    cards_played: dict[str, int]


@dataclass(frozen=True)
class MultiPolicyEvaluationResult:
    episodes: int
    players: int
    seat_results: tuple[SeatEvaluationResult, ...]
    episode_results: tuple[MultiPolicyEpisodeResult, ...]
    trained_seats: tuple[str, ...]
    trained_combined_wins: int
    trained_combined_win_rate: float
    random_seats: tuple[str, ...]
    random_combined_wins: int
    random_combined_win_rate: float


class ModelController:
    is_trained = True

    def __init__(self, model: Any, policy_label: str) -> None:
        self.model = model
        self.policy_label = policy_label

    def choose_action(
        self,
        observation: GameObservation,
        encoded_observation: dict[str, np.ndarray],
        action_encoder: ActionEncoder,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ) -> GameAction:
        del np_rng, py_rng
        action, _ = self.model.predict(
            encoded_observation,
            deterministic=True,
            action_masks=encoded_observation["action_mask"].astype(bool),
        )
        return action_encoder.decode(int(action), observation)


class RandomMaskController:
    policy_label = "random"
    is_trained = False

    def choose_action(
        self,
        observation: GameObservation,
        encoded_observation: dict[str, np.ndarray],
        action_encoder: ActionEncoder,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ) -> GameAction:
        del py_rng
        legal_action_ids = np.flatnonzero(encoded_observation["action_mask"])
        if len(legal_action_ids) == 0:
            raise ValueError(f"{observation.player} has no legal actions.")
        return action_encoder.decode(int(np_rng.choice(legal_action_ids)), observation)


class ScriptedAgentController:
    is_trained = False

    def __init__(self, policy_label: str) -> None:
        self.policy_label = policy_label
        agent_types = {
            "draw-only": DrawOnlyAgent,
            "random": RandomAgent,
            "safe-rule": SafeRuleAgent,
            "skip-if-possible": SkipIfPossibleAgent,
        }
        self.agent = agent_types[policy_label]()

    def choose_action(
        self,
        observation: GameObservation,
        encoded_observation: dict[str, np.ndarray],
        action_encoder: ActionEncoder,
        np_rng: np.random.Generator,
        py_rng: random.Random,
    ) -> GameAction:
        del encoded_observation, action_encoder, np_rng
        return self.agent.choose_action(observation, py_rng)


def evaluate_multi_policy_game(
    *,
    seat_policies: dict[str, str],
    episodes: int = 100,
    players: int = 3,
    seed: int = 1,
    max_turns: int = 500,
    include_cards: tuple[str, ...] | None = None,
    exclude_cards: tuple[str, ...] = (),
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
    max_hand_size: int = 32,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> MultiPolicyEvaluationResult:
    validate_evaluation_args(
        seat_policies=seat_policies,
        episodes=episodes,
        players=players,
        wandb_mode=wandb_mode,
    )
    player_names = tuple(f"player_{index + 1}" for index in range(players))
    completed_specs = complete_policy_specs(seat_policies, player_names)
    controllers = build_controllers(completed_specs)
    config = {
        "episodes": episodes,
        "players": players,
        "seat_policies": {spec.seat: spec.policy for spec in completed_specs.values()},
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "max_hand_size": max_hand_size,
    }
    run = start_wandb_run(
        mode=wandb_mode,
        project=wandb_project,
        entity=wandb_entity,
        config=config,
    )
    try:
        episode_results = tuple(
            run_multi_policy_episode(
                episode=episode_index,
                player_names=player_names,
                controllers=controllers,
                seed=seed + episode_index,
                max_turns=max_turns,
                include_cards=include_cards,
                exclude_cards=exclude_cards,
                enabled_combo_rules=enabled_combo_rules,
                reveal_opponent_card_counts=reveal_opponent_card_counts,
                max_hand_size=max_hand_size,
            )
            for episode_index in range(episodes)
        )
        result = summarize_evaluation(
            episode_results=episode_results,
            controllers=controllers,
            player_names=player_names,
        )
        log_results(run, result)
        return result
    finally:
        if run is not None:
            run.finish()


def run_multi_policy_episode(
    *,
    episode: int,
    player_names: tuple[str, ...],
    controllers: dict[str, SeatController],
    seed: int,
    max_turns: int,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...],
    reveal_opponent_card_counts: bool,
    max_hand_size: int = 32,
) -> MultiPolicyEpisodeResult:
    engine = GameEngine.new_game(
        player_names,
        seed=seed,
        include_cards=include_cards,
        exclude_cards=exclude_cards,
        enabled_combo_rules=enabled_combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )
    action_encoder = ActionEncoder(
        max_hand_size=max_hand_size,
        max_players=len(player_names),
        player_names=player_names,
    )
    observation_encoder = ObservationEncoder(
        max_hand_size=max_hand_size,
        max_players=len(player_names),
        player_names=player_names,
        include_opponent_card_counts=reveal_opponent_card_counts,
        action_encoder=action_encoder,
    )
    np_rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    event_cursor = 0
    rewards = {player: 0.0 for player in player_names}
    turns_survived = {player: 0 for player in player_names}
    defuses = Counter({player: 0 for player in player_names})
    explosions = Counter({player: 0 for player in player_names})
    cards_played: dict[str, Counter[str]] = {
        player: Counter() for player in player_names
    }

    while not engine.is_over and engine.turn_count < max_turns:
        observation = engine.observe(engine.current_player.name)
        encoded_observation = observation_encoder.encode(observation)
        action = controllers[observation.player].choose_action(
            observation,
            encoded_observation,
            action_encoder,
            np_rng,
            py_rng,
        )
        engine.step(action)

        new_events = engine.events[event_cursor:]
        event_cursor = len(engine.events)
        for event in new_events:
            if event.kind == "defuse":
                defuses[event.player] += 1
            if event.kind == "eliminated":
                explosions[event.player] += 1
                rewards[event.player] -= 1.0
                turns_survived[event.player] = engine.turn_count
            if event.kind in COUNTED_CARD_EVENTS:
                cards_played[event.player][event.kind] += 1

    truncated = not engine.is_over and engine.turn_count >= max_turns
    winner = engine.winner
    if winner is not None:
        rewards[winner] += 1.0
    for player in player_names:
        if turns_survived[player] == 0:
            turns_survived[player] = engine.turn_count

    return MultiPolicyEpisodeResult(
        episode=episode,
        seed=seed,
        winner=winner,
        truncated=truncated,
        turns=engine.turn_count,
        rewards=rewards,
        turns_survived=turns_survived,
        defuses=dict(defuses),
        explosions=dict(explosions),
        cards_played={
            player: dict(sorted(counter.items()))
            for player, counter in cards_played.items()
        },
    )


def summarize_evaluation(
    *,
    episode_results: tuple[MultiPolicyEpisodeResult, ...],
    controllers: dict[str, SeatController],
    player_names: tuple[str, ...],
) -> MultiPolicyEvaluationResult:
    episodes = len(episode_results)
    seat_results: list[SeatEvaluationResult] = []
    for seat in player_names:
        cards_played: Counter[str] = Counter()
        for episode in episode_results:
            cards_played.update(episode.cards_played[seat])
        wins = sum(episode.winner == seat for episode in episode_results)
        seat_results.append(
            SeatEvaluationResult(
                seat=seat,
                policy_label=controllers[seat].policy_label,
                wins=wins,
                win_rate=wins / episodes,
                average_reward=float(
                    np.mean([episode.rewards[seat] for episode in episode_results])
                ),
                average_turns_survived=float(
                    np.mean(
                        [episode.turns_survived[seat] for episode in episode_results]
                    )
                ),
                defuses=sum(episode.defuses[seat] for episode in episode_results),
                explosions=sum(episode.explosions[seat] for episode in episode_results),
                cards_played=dict(sorted(cards_played.items())),
            )
        )

    trained_seats = tuple(
        seat for seat in player_names if controllers[seat].is_trained
    )
    random_seats = tuple(
        seat for seat in player_names if controllers[seat].policy_label == "random"
    )
    trained_combined_wins = sum(
        episode.winner in trained_seats for episode in episode_results
    )
    random_combined_wins = sum(
        episode.winner in random_seats for episode in episode_results
    )
    return MultiPolicyEvaluationResult(
        episodes=episodes,
        players=len(player_names),
        seat_results=tuple(seat_results),
        episode_results=episode_results,
        trained_seats=trained_seats,
        trained_combined_wins=trained_combined_wins,
        trained_combined_win_rate=trained_combined_wins / episodes,
        random_seats=random_seats,
        random_combined_wins=random_combined_wins,
        random_combined_win_rate=random_combined_wins / episodes,
    )


def complete_policy_specs(
    seat_policies: dict[str, str],
    player_names: tuple[str, ...],
) -> dict[str, SeatPolicySpec]:
    unknown_seats = sorted(set(seat_policies) - set(player_names))
    if unknown_seats:
        raise ValueError(f"Unknown seat(s): {', '.join(unknown_seats)}.")
    return {
        player_name: SeatPolicySpec(
            seat=player_name,
            policy=seat_policies.get(player_name, "random"),
        )
        for player_name in player_names
    }


def build_controllers(
    policy_specs: dict[str, SeatPolicySpec],
) -> dict[str, SeatController]:
    return {
        seat: build_controller(spec)
        for seat, spec in policy_specs.items()
    }


def build_controller(spec: SeatPolicySpec) -> SeatController:
    if spec.is_model_policy:
        model_path = spec.model_path
        if model_path is None or not model_path.exists():
            raise ValueError(f"Model file does not exist for {spec.seat}: {model_path}")
        model = load_maskable_ppo(model_path)
        return ModelController(model=model, policy_label=spec.label)
    if spec.policy == "random":
        return RandomMaskController()
    if spec.policy in SCRIPTED_POLICY_KINDS:
        return ScriptedAgentController(spec.policy)
    known = ", ".join(("model:<path>", *SCRIPTED_POLICY_KINDS))
    raise ValueError(f"Unknown policy for {spec.seat}: {spec.policy}. Known: {known}.")


def validate_evaluation_args(
    *,
    seat_policies: dict[str, str],
    episodes: int,
    players: int,
    wandb_mode: str,
) -> None:
    if players < 2:
        raise ValueError("players must be at least 2.")
    if episodes < 1:
        raise ValueError("episodes must be at least 1.")
    if not seat_policies:
        raise ValueError("At least one seat policy is required.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


def load_maskable_ppo(model_path: Path) -> Any:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError(
            'Multi-policy model evaluation requires `pip install -e ".[training]"` first.'
        ) from exc
    return MaskablePPO.load(model_path)


def parse_seat_policies(value: str) -> dict[str, str]:
    policies: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                "Seat policies must look like player_1=model:path.zip,player_2=random."
            )
        seat, policy = item.split("=", 1)
        seat = seat.strip()
        policy = policy.strip()
        if not seat or not policy:
            raise argparse.ArgumentTypeError("Seat and policy values cannot be empty.")
        policies[seat] = policy
    return policies


def parse_name_list(value: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in value.split(",") if name.strip())


def parse_optional_name_list(value: str) -> tuple[str, ...] | None:
    names = parse_name_list(value)
    return names or None


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
        job_type="multi-policy-evaluation",
    )


def log_results(run: Any | None, result: MultiPolicyEvaluationResult) -> None:
    if run is None:
        return
    run.log(
        {
            "multi_policy/trained_combined_win_rate": result.trained_combined_win_rate,
            "multi_policy/random_combined_win_rate": result.random_combined_win_rate,
            "multi_policy/episodes": result.episodes,
        }
    )
    for seat_result in result.seat_results:
        prefix = f"multi_policy/{seat_result.seat}"
        run.log(
            {
                f"{prefix}/win_rate": seat_result.win_rate,
                f"{prefix}/average_reward": seat_result.average_reward,
                f"{prefix}/average_turns_survived": seat_result.average_turns_survived,
                f"{prefix}/defuses": seat_result.defuses,
                f"{prefix}/explosions": seat_result.explosions,
            }
        )
    log_results_table(run, result)


def log_results_table(run: Any | None, result: MultiPolicyEvaluationResult) -> None:
    if run is None:
        return
    import wandb

    table = wandb.Table(
        columns=[
            "seat",
            "policy",
            "wins",
            "win_rate",
            "average_reward",
            "average_turns_survived",
            "defuses",
            "explosions",
            "cards_played",
        ]
    )
    for seat_result in result.seat_results:
        table.add_data(
            seat_result.seat,
            seat_result.policy_label,
            seat_result.wins,
            seat_result.win_rate,
            seat_result.average_reward,
            seat_result.average_turns_survived,
            seat_result.defuses,
            seat_result.explosions,
            json.dumps(seat_result.cards_played, sort_keys=True),
        )
    run.log({"multi_policy/seat_comparison": table})


def format_results_table(result: MultiPolicyEvaluationResult) -> str:
    headers = (
        "seat",
        "policy",
        "wins",
        "win_rate",
        "avg_reward",
        "avg_turns",
        "explosions",
    )
    rows = [
        (
            seat_result.seat,
            seat_result.policy_label,
            str(seat_result.wins),
            f"{seat_result.win_rate:.3f}",
            f"{seat_result.average_reward:.3f}",
            f"{seat_result.average_turns_survived:.1f}",
            str(seat_result.explosions),
        )
        for seat_result in result.seat_results
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
    lines.append(
        f"trained_combined_win_rate={result.trained_combined_win_rate:.3f} "
        f"random_combined_win_rate={result.random_combined_win_rate:.3f}"
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate different policies sharing one multi-player game."
    )
    parser.add_argument("--seat-policies", type=parse_seat_policies, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument("--include-cards", type=parse_optional_name_list, default=None)
    parser.add_argument("--exclude-cards", type=parse_name_list, default=())
    parser.add_argument("--enabled-combo-rules", type=parse_name_list, default=())
    parser.add_argument("--reveal-opponent-card-counts", action="store_true")
    parser.add_argument("--max-hand-size", type=int, default=32)
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    parser.add_argument("--wandb-project", default="exploding-kittens")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_multi_policy_game(
        seat_policies=args.seat_policies,
        episodes=args.episodes,
        players=args.players,
        seed=args.seed,
        max_turns=args.max_turns,
        include_cards=args.include_cards,
        exclude_cards=args.exclude_cards,
        enabled_combo_rules=args.enabled_combo_rules,
        reveal_opponent_card_counts=args.reveal_opponent_card_counts,
        max_hand_size=args.max_hand_size,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("Multi-Policy Evaluation")
    print(f"Episodes: {result.episodes}")
    print(f"Players: {result.players}")
    print(format_results_table(result))


if __name__ == "__main__":
    main()
