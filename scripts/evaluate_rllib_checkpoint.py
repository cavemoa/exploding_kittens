from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploding_kittens import ActionEncoder, GameAction, GameObservation
from scripts.evaluate_multi_policy_game import (
    SCRIPTED_POLICY_KINDS,
    MultiPolicyEvaluationResult,
    RandomMaskController,
    ScriptedAgentController,
    SeatController,
    SeatEvaluationResult,
    complete_policy_specs,
    format_results_table as format_base_results_table,
    parse_name_list,
    parse_optional_name_list,
    parse_seat_policies,
    run_multi_policy_episode,
    summarize_evaluation,
)
from scripts.train_rllib_pettingzoo import (
    DEFAULT_MODEL_NAME,
    SHARED_POLICY_ID,
    register_action_mask_model,
)


WANDB_MODES = ("online", "offline", "disabled")


@dataclass(frozen=True)
class RLLibCheckpointEvaluationResult:
    base_result: MultiPolicyEvaluationResult
    rllib_seats: tuple[str, ...]
    rllib_combined_wins: int
    rllib_combined_win_rate: float
    scripted_opponent_seats: tuple[str, ...]
    scripted_opponent_wins: int
    scripted_opponent_win_rate: float
    seat_order_advantage: float
    action_distribution: dict[str, int] = field(default_factory=dict)
    card_distribution: dict[str, int] = field(default_factory=dict)
    combo_count: int = 0
    illegal_action_rate: float = 0.0


class RLLibController:
    is_trained = True

    def __init__(
        self,
        *,
        policy: Any,
        policy_label: str,
        explore: bool = False,
    ) -> None:
        self.policy = policy
        self.policy_label = policy_label
        self.explore = explore

    def choose_action(
        self,
        observation: GameObservation,
        encoded_observation: dict[str, np.ndarray],
        action_encoder: ActionEncoder,
        np_rng: np.random.Generator,
        py_rng: Any,
    ) -> GameAction:
        del np_rng, py_rng
        rllib_observation = {
            "observations": encoded_observation["observation"],
            "action_mask": encoded_observation["action_mask"],
        }
        action_id, _, _ = self.policy.compute_single_action(
            rllib_observation,
            explore=self.explore,
        )
        action_id = int(action_id)
        if not encoded_observation["action_mask"][action_id]:
            raise ValueError(
                f"RLlib policy selected illegal action ID {action_id} "
                f"for {observation.player}."
            )
        return action_encoder.decode(action_id, observation)


def evaluate_rllib_checkpoint(
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
    policy_id: str = SHARED_POLICY_ID,
    explore: bool = False,
    wandb_mode: str = "disabled",
    wandb_project: str = "exploding-kittens",
    wandb_entity: str | None = None,
) -> RLLibCheckpointEvaluationResult:
    validate_evaluation_args(
        seat_policies=seat_policies,
        episodes=episodes,
        players=players,
        wandb_mode=wandb_mode,
    )
    player_names = tuple(f"player_{index + 1}" for index in range(players))
    completed_specs = complete_policy_specs(seat_policies, player_names)
    controllers = build_controllers(
        {seat: spec.policy for seat, spec in completed_specs.items()},
        policy_id=policy_id,
        explore=explore,
    )
    config = {
        "episodes": episodes,
        "players": players,
        "seat_policies": {seat: spec.policy for seat, spec in completed_specs.items()},
        "seed": seed,
        "max_turns": max_turns,
        "include_cards": include_cards,
        "exclude_cards": exclude_cards,
        "enabled_combo_rules": enabled_combo_rules,
        "reveal_opponent_card_counts": reveal_opponent_card_counts,
        "max_hand_size": max_hand_size,
        "policy_id": policy_id,
        "explore": explore,
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
        base_result = summarize_evaluation(
            episode_results=episode_results,
            controllers=controllers,
            player_names=player_names,
        )
        result = summarize_rllib_evaluation(
            base_result=base_result,
            controllers=controllers,
            player_names=player_names,
        )
        log_results(run, result)
        return result
    finally:
        if run is not None:
            run.finish()


def build_controllers(
    seat_policies: dict[str, str],
    *,
    policy_id: str = SHARED_POLICY_ID,
    explore: bool = False,
) -> dict[str, SeatController]:
    loaded_policies: dict[Path, Any] = {}
    controllers: dict[str, SeatController] = {}
    for seat, policy_spec in seat_policies.items():
        if policy_spec.startswith("rllib:"):
            checkpoint_path = Path(policy_spec.removeprefix("rllib:"))
            if checkpoint_path not in loaded_policies:
                loaded_policies[checkpoint_path] = load_rllib_policy(
                    checkpoint_path,
                    policy_id=policy_id,
                )
            controllers[seat] = RLLibController(
                policy=loaded_policies[checkpoint_path],
                policy_label=f"rllib:{checkpoint_path.name}",
                explore=explore,
            )
        elif policy_spec == "random":
            controllers[seat] = RandomMaskController()
        elif policy_spec in SCRIPTED_POLICY_KINDS:
            controllers[seat] = ScriptedAgentController(policy_spec)
        else:
            known = ", ".join(("rllib:<checkpoint_path>", *SCRIPTED_POLICY_KINDS))
            raise ValueError(
                f"Unknown policy for {seat}: {policy_spec}. Known: {known}."
            )
    return controllers


def load_rllib_policy(checkpoint_path: Path, *, policy_id: str) -> Any:
    if not checkpoint_path.exists():
        raise ValueError(f"RLlib checkpoint does not exist: {checkpoint_path}")
    register_action_mask_model(DEFAULT_MODEL_NAME)
    try:
        from ray.rllib.policy.policy import Policy
    except ImportError as exc:
        raise RuntimeError(
            'RLlib checkpoint evaluation requires `pip install -e ".[multiagent]"` first.'
        ) from exc

    restored = Policy.from_checkpoint(
        str(checkpoint_path.resolve()),
        policy_ids=[policy_id],
    )
    if not isinstance(restored, dict):
        return restored
    try:
        return restored[policy_id]
    except KeyError as exc:
        available = ", ".join(restored)
        raise ValueError(
            f"Policy ID {policy_id!r} was not found in {checkpoint_path}. "
            f"Available policies: {available}."
        ) from exc


def summarize_rllib_evaluation(
    *,
    base_result: MultiPolicyEvaluationResult,
    controllers: dict[str, SeatController],
    player_names: tuple[str, ...],
) -> RLLibCheckpointEvaluationResult:
    rllib_seats = tuple(
        seat
        for seat in player_names
        if isinstance(controllers[seat], RLLibController)
    )
    scripted_opponent_seats = tuple(
        seat
        for seat in player_names
        if not isinstance(controllers[seat], RLLibController)
    )
    rllib_combined_wins = sum(
        episode.winner in rllib_seats
        for episode in base_result.episode_results
    )
    scripted_opponent_wins = sum(
        episode.winner in scripted_opponent_seats
        for episode in base_result.episode_results
    )
    seat_win_rates = [seat_result.win_rate for seat_result in base_result.seat_results]
    seat_order_advantage = max(seat_win_rates) - min(seat_win_rates)
    action_distribution: Counter[str] = Counter()
    card_distribution: Counter[str] = Counter()
    for seat in rllib_seats:
        seat_result = next(
            result for result in base_result.seat_results if result.seat == seat
        )
        action_distribution.update(seat_result.action_counts)
        card_distribution.update(seat_result.cards_played)
    return RLLibCheckpointEvaluationResult(
        base_result=base_result,
        rllib_seats=rllib_seats,
        rllib_combined_wins=rllib_combined_wins,
        rllib_combined_win_rate=rllib_combined_wins / base_result.episodes,
        scripted_opponent_seats=scripted_opponent_seats,
        scripted_opponent_wins=scripted_opponent_wins,
        scripted_opponent_win_rate=scripted_opponent_wins / base_result.episodes,
        seat_order_advantage=seat_order_advantage,
        action_distribution=dict(sorted(action_distribution.items())),
        card_distribution=dict(sorted(card_distribution.items())),
        combo_count=card_distribution.get("two_of_a_kind", 0),
        illegal_action_rate=0.0,
    )


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
    if not any(policy.startswith("rllib:") for policy in seat_policies.values()):
        raise ValueError("At least one seat policy must use rllib:<checkpoint_path>.")
    if wandb_mode not in WANDB_MODES:
        raise ValueError(f"wandb_mode must be one of: {', '.join(WANDB_MODES)}.")


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
        job_type="rllib-checkpoint-evaluation",
    )


def log_results(
    run: Any | None,
    result: RLLibCheckpointEvaluationResult,
) -> None:
    if run is None:
        return
    run.log(
        {
            "rllib_eval/episodes": result.base_result.episodes,
            "rllib_eval/rllib_combined_win_rate": result.rllib_combined_win_rate,
            "rllib_eval/scripted_opponent_win_rate": (
                result.scripted_opponent_win_rate
            ),
            "rllib_eval/seat_order_advantage": result.seat_order_advantage,
            "rllib_eval/illegal_action_rate": result.illegal_action_rate,
            "rllib_eval/combo_count": result.combo_count,
        }
    )
    for action, count in result.action_distribution.items():
        run.log({f"rllib_eval/action_distribution/{action}": count})
    for card, count in result.card_distribution.items():
        run.log({f"rllib_eval/card_distribution/{card}": count})
    for seat_result in result.base_result.seat_results:
        prefix = f"rllib_eval/{seat_result.seat}"
        run.log(
            {
                f"{prefix}/win_rate": seat_result.win_rate,
                f"{prefix}/average_reward": seat_result.average_reward,
                f"{prefix}/average_turns_survived": (
                    seat_result.average_turns_survived
                ),
                f"{prefix}/defuses": seat_result.defuses,
                f"{prefix}/explosions": seat_result.explosions,
            }
        )
    log_results_table(run, result)


def log_results_table(
    run: Any | None,
    result: RLLibCheckpointEvaluationResult,
) -> None:
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
            "action_counts",
        ]
    )
    for seat_result in result.base_result.seat_results:
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
            json.dumps(seat_result.action_counts, sort_keys=True),
        )
    run.log({"rllib_eval/seat_comparison": table})


def format_results_table(result: RLLibCheckpointEvaluationResult) -> str:
    base_table = format_base_results_table(result.base_result)
    return "\n".join(
        (
            base_table,
            f"rllib_combined_win_rate={result.rllib_combined_win_rate:.3f}",
            f"scripted_opponent_win_rate={result.scripted_opponent_win_rate:.3f}",
            f"seat_order_advantage={result.seat_order_advantage:.3f}",
            f"illegal_action_rate={result.illegal_action_rate:.3f}",
            f"combo_count={result.combo_count}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RLlib checkpoints against scripted opponents."
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
    parser.add_argument("--policy-id", default=SHARED_POLICY_ID)
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--wandb-mode", choices=WANDB_MODES, default="disabled")
    parser.add_argument("--wandb-project", default="exploding-kittens")
    parser.add_argument("--wandb-entity", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate_rllib_checkpoint(
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
        policy_id=args.policy_id,
        explore=args.explore,
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )

    print("RLlib Checkpoint Evaluation")
    print(f"Episodes: {result.base_result.episodes}")
    print(f"Players: {result.base_result.players}")
    print(format_results_table(result))


if __name__ == "__main__":
    main()
