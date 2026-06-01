from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
import random

import yaml

from exploding_kittens.actions import Action, GameAction
from exploding_kittens import (
    DrawOnlyAgent,
    GameEngine,
    RandomAgent,
    SafeRuleAgent,
    SkipIfPossibleAgent,
)
from exploding_kittens.cards import DECK_CARD_REGISTRY
from exploding_kittens.combo_rules import COMBO_RULE_REGISTRY, ComboRule


DEFAULT_CONFIG_PATH = Path("game_config.yaml")

AGENTS = {
    "draw-only": DrawOnlyAgent,
    "random": RandomAgent,
    "safe-rule": SafeRuleAgent,
    "skip-if-possible": SkipIfPossibleAgent,
}


@dataclass(frozen=True)
class SimulationConfig:
    games: int = 1000
    players: int = 2
    strategy: str = "random"
    seed: int = 1
    max_turns: int = 500
    verbose: bool = False
    include_cards: tuple[str, ...] | None = None
    exclude_cards: tuple[str, ...] = ()
    enabled_combo_rules: tuple[str, ...] = ()
    reveal_opponent_card_counts: bool = False


@dataclass(frozen=True)
class GameResult:
    winner: str | None
    turns: int
    event_counts: Counter[str]
    agent_stats: dict[str, AgentGameStats]


@dataclass
class AgentGameStats:
    turns_survived: int = 0
    cards_played: Counter[str] = field(default_factory=Counter)
    combos_played: Counter[str] = field(default_factory=Counter)
    explosions: int = 0
    defuses: int = 0
    hand_size_samples: list[int] = field(default_factory=list)


@dataclass
class AggregateAgentStats:
    games: int = 0
    wins: int = 0
    turns_survived: int = 0
    cards_played: Counter[str] = field(default_factory=Counter)
    combos_played: Counter[str] = field(default_factory=Counter)
    explosions: int = 0
    defuses: int = 0
    hand_size_total: int = 0
    hand_size_samples: int = 0


def play_game(
    player_count: int,
    strategy: str,
    seed: int,
    max_turns: int,
    verbose: bool,
    include_cards: tuple[str, ...] | None,
    exclude_cards: tuple[str, ...],
    enabled_combo_rules: tuple[str, ...] = (),
    reveal_opponent_card_counts: bool = False,
) -> GameResult:
    player_names = [f"player_{index + 1}" for index in range(player_count)]
    engine = GameEngine.new_game(
        player_names,
        seed=seed,
        include_cards=include_cards,
        exclude_cards=exclude_cards,
        enabled_combo_rules=enabled_combo_rules,
        reveal_opponent_card_counts=reveal_opponent_card_counts,
    )
    agents = {name: AGENTS[strategy]() for name in player_names}
    rng = random.Random(seed)
    agent_stats = {name: AgentGameStats() for name in player_names}
    elimination_turns: dict[str, int] = {}

    sample_hand_sizes(engine, agent_stats)
    while not engine.is_over and engine.turn_count < max_turns:
        player = engine.current_player.name
        observation = engine.observe(player)
        action = agents[player].choose_action(observation, rng)
        record_chosen_action(engine, action, agent_stats[player])
        events = engine.step(action)
        for event in events:
            if event.kind == "defuse":
                agent_stats[event.player].defuses += 1
            elif event.kind == "eliminated":
                agent_stats[event.player].explosions += 1
                elimination_turns[event.player] = engine.turn_count
        sample_hand_sizes(engine, agent_stats)
        if verbose:
            for event in events:
                print(event.message)

    for player_name, stats in agent_stats.items():
        stats.turns_survived = elimination_turns.get(player_name, engine.turn_count)

    return GameResult(
        winner=engine.winner,
        turns=engine.turn_count,
        event_counts=Counter(event.kind for event in engine.events),
        agent_stats=agent_stats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simplified Exploding Kittens games.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--games", type=int)
    parser.add_argument("--players", type=int)
    parser.add_argument("--strategy", choices=sorted(AGENTS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--verbose", action="store_true", default=None)
    parser.add_argument("--include-cards", type=parse_card_list)
    parser.add_argument("--exclude-cards", type=parse_card_list)
    parser.add_argument("--enabled-combo-rules", type=parse_card_list)
    parser.add_argument("--reveal-opponent-card-counts", action="store_true", default=None)
    return parser.parse_args()


def parse_card_list(value: str) -> tuple[str, ...]:
    cards = tuple(card.strip() for card in value.split(",") if card.strip())
    if not cards:
        raise argparse.ArgumentTypeError("card list cannot be empty.")
    return cards


def sample_hand_sizes(
    engine: GameEngine,
    agent_stats: dict[str, AgentGameStats],
) -> None:
    for player in engine.players:
        if player.alive:
            agent_stats[player.name].hand_size_samples.append(len(player.hand))


def record_chosen_action(
    engine: GameEngine,
    action: GameAction | Action | str,
    stats: AgentGameStats,
) -> None:
    game_action = normalize_game_action(action)
    if game_action.kind is Action.DRAW:
        return

    for card_index in game_action.card_indexes:
        if 0 <= card_index < len(engine.current_player.hand):
            card = engine.current_player.hand[card_index]
            stats.cards_played[card.name] += 1

    combo_name = combo_name_for_action(engine.combo_rules, game_action)
    if combo_name is not None:
        stats.combos_played[combo_name] += 1


def normalize_game_action(action: GameAction | Action | str) -> GameAction:
    if isinstance(action, GameAction):
        return action
    return GameAction(kind=Action(action))


def combo_name_for_action(
    combo_rules: list[ComboRule],
    action: GameAction,
) -> str | None:
    return next(
        (combo_rule.name for combo_rule in combo_rules if combo_rule.action is action.kind),
        None,
    )


def load_yaml_config(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    allowed_keys = {field.name for field in fields(SimulationConfig)}
    unknown_keys = sorted(set(loaded) - allowed_keys)
    if unknown_keys:
        keys = ", ".join(unknown_keys)
        raise ValueError(f"{path} contains unknown setting(s): {keys}")

    return loaded


def build_config(args: argparse.Namespace) -> SimulationConfig:
    values = vars(SimulationConfig())
    values.update(load_yaml_config(args.config))

    for field in fields(SimulationConfig):
        cli_value = getattr(args, field.name, None)
        if cli_value is not None:
            values[field.name] = cli_value

    values["include_cards"] = normalize_card_list(values["include_cards"], "include_cards")
    values["exclude_cards"] = normalize_card_list(values["exclude_cards"], "exclude_cards") or ()
    values["enabled_combo_rules"] = (
        normalize_card_list(values["enabled_combo_rules"], "enabled_combo_rules") or ()
    )

    config = SimulationConfig(**values)
    validate_config(config)
    return config


def normalize_card_list(value: object, setting_name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return parse_card_list(value)
    if isinstance(value, list | tuple):
        cards = tuple(value)
        if not all(isinstance(card, str) for card in cards):
            raise ValueError(f"{setting_name} must contain only strings.")
        return cards
    raise ValueError(f"{setting_name} must be a list of card names.")


def validate_config(config: SimulationConfig) -> None:
    if config.games < 1:
        raise ValueError("games must be at least 1.")
    if config.players < 2:
        raise ValueError("players must be at least 2.")
    if config.strategy not in AGENTS:
        choices = ", ".join(sorted(AGENTS))
        raise ValueError(f"strategy must be one of: {choices}.")
    if config.max_turns < 1:
        raise ValueError("max_turns must be at least 1.")
    known_cards = set(DECK_CARD_REGISTRY)
    configured_cards = set(config.exclude_cards)
    if config.include_cards is not None:
        configured_cards.update(config.include_cards)
    unknown_cards = sorted(configured_cards - known_cards)
    if unknown_cards:
        known = ", ".join(sorted(known_cards))
        unknown = ", ".join(unknown_cards)
        raise ValueError(f"Unknown optional card(s): {unknown}. Known optional cards: {known}.")
    if config.include_cards is None:
        active_cards = known_cards - set(config.exclude_cards)
    else:
        active_cards = set(config.include_cards) - set(config.exclude_cards)
    if not active_cards:
        raise ValueError("include_cards and exclude_cards leave no optional deck cards.")
    unknown_combo_rules = sorted(set(config.enabled_combo_rules) - set(COMBO_RULE_REGISTRY))
    if unknown_combo_rules:
        known = ", ".join(sorted(COMBO_RULE_REGISTRY))
        unknown = ", ".join(unknown_combo_rules)
        raise ValueError(f"Unknown combo rule(s): {unknown}. Known combo rules: {known}.")


def main() -> None:
    args = parse_args()
    config = build_config(args)

    wins: Counter[str | None] = Counter()
    event_totals: Counter[str] = Counter()
    all_turns: list[int] = []
    completed_turns: list[int] = []
    total_turns = 0
    agent_totals = {
        f"player_{index + 1}": AggregateAgentStats()
        for index in range(config.players)
    }
    for game_index in range(config.games):
        result = play_game(
            player_count=config.players,
            strategy=config.strategy,
            seed=config.seed + game_index,
            max_turns=config.max_turns,
            verbose=config.verbose and game_index == 0,
            include_cards=config.include_cards,
            exclude_cards=config.exclude_cards,
            enabled_combo_rules=config.enabled_combo_rules,
            reveal_opponent_card_counts=config.reveal_opponent_card_counts,
        )
        wins[result.winner] += 1
        event_totals.update(result.event_counts)
        all_turns.append(result.turns)
        total_turns += result.turns
        if result.winner is not None:
            completed_turns.append(result.turns)
        update_agent_totals(agent_totals, result)

    print(f"Config: {args.config if args.config.exists() else 'built-in defaults'}")
    print(f"Games: {config.games}")
    print(f"Players: {config.players}")
    print(f"Strategy: {config.strategy}")
    print(f"Cards: {format_active_cards(config)}")
    print(f"Combo rules: {format_enabled_combo_rules(config)}")
    print()
    print("Results")
    for winner, count in wins.most_common():
        label = winner if winner is not None else "no winner"
        print(f"{label}: {count} ({count / config.games:.1%})")
    print()
    print("Diagnostics")
    print(f"Finished games: {len(completed_turns)} ({len(completed_turns) / config.games:.1%})")
    print(f"No winner: {wins[None]} ({wins[None] / config.games:.1%})")
    print(f"Average turns: {total_turns / config.games:.2f}")
    if completed_turns:
        print(f"Average turns to win: {sum(completed_turns) / len(completed_turns):.2f}")
    print(f"Shortest game: {min(all_turns)} turns")
    print(f"Longest game: {max(all_turns)} turns")
    print_event_diagnostics(event_totals, config.games)
    print_agent_diagnostics(agent_totals)


def format_active_cards(config: SimulationConfig) -> str:
    if config.include_cards is None:
        active_cards = set(DECK_CARD_REGISTRY)
    else:
        active_cards = set(config.include_cards)
    active_cards -= set(config.exclude_cards)
    return ", ".join(sorted(active_cards))


def print_event_diagnostics(event_totals: Counter[str], games: int) -> None:
    event_labels = {
        "draw": "Draws",
        "skip": "Skips played",
        "attack": "Attacks played",
        "shuffle": "Shuffles played",
        "favor": "Favors played",
        "alter_the_future": "Alter the Future played",
        "see_the_future": "See the Future played",
        "two_of_a_kind": "Two of a Kind played",
        "defuse": "Defuses used",
        "eliminated": "Eliminations",
    }
    for event_kind, label in event_labels.items():
        count = event_totals[event_kind]
        print(f"{label}: {count} ({count / games:.2f}/game)")


def update_agent_totals(
    agent_totals: dict[str, AggregateAgentStats],
    result: GameResult,
) -> None:
    for player_name, game_stats in result.agent_stats.items():
        totals = agent_totals[player_name]
        totals.games += 1
        totals.wins += int(result.winner == player_name)
        totals.turns_survived += game_stats.turns_survived
        totals.cards_played.update(game_stats.cards_played)
        totals.combos_played.update(game_stats.combos_played)
        totals.explosions += game_stats.explosions
        totals.defuses += game_stats.defuses
        totals.hand_size_total += sum(game_stats.hand_size_samples)
        totals.hand_size_samples += len(game_stats.hand_size_samples)


def print_agent_diagnostics(agent_totals: dict[str, AggregateAgentStats]) -> None:
    print()
    print("Per-agent Diagnostics")
    player_names = tuple(sorted(agent_totals))

    print_table(
        "Summary",
        "Metric",
        player_names,
        build_agent_summary_rows(agent_totals, player_names),
    )
    print()
    print_table(
        "Cards Played / Game",
        "Card",
        player_names,
        build_counter_rate_rows(agent_totals, player_names, "cards_played"),
    )
    print()
    print_table(
        "Combos Played / Game",
        "Combo",
        player_names,
        build_counter_rate_rows(agent_totals, player_names, "combos_played"),
    )


def build_agent_summary_rows(
    agent_totals: dict[str, AggregateAgentStats],
    player_names: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return (
        (
            "Win rate",
            tuple(format_rate(agent_totals[player].wins, agent_totals[player].games) for player in player_names),
        ),
        (
            "Avg turns survived",
            tuple(format_average(agent_totals[player].turns_survived, agent_totals[player].games) for player in player_names),
        ),
        (
            "Avg hand size",
            tuple(
                format_average(
                    agent_totals[player].hand_size_total,
                    agent_totals[player].hand_size_samples,
                )
                for player in player_names
            ),
        ),
        (
            "Explosions/game",
            tuple(format_average(agent_totals[player].explosions, agent_totals[player].games) for player in player_names),
        ),
        (
            "Defuses/game",
            tuple(format_average(agent_totals[player].defuses, agent_totals[player].games) for player in player_names),
        ),
    )


def build_counter_rate_rows(
    agent_totals: dict[str, AggregateAgentStats],
    player_names: tuple[str, ...],
    counter_name: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    names = sorted(
        {
            name
            for player in player_names
            for name in getattr(agent_totals[player], counter_name)
        }
    )
    if not names:
        return (("none", tuple("-" for _ in player_names)),)
    return tuple(
        (
            name,
            tuple(
                format_average(
                    getattr(agent_totals[player], counter_name)[name],
                    agent_totals[player].games,
                )
                for player in player_names
            ),
        )
        for name in names
    )


def print_table(
    title: str,
    row_header: str,
    column_headers: tuple[str, ...],
    rows: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    print(title)
    for line in build_ascii_table(row_header, column_headers, rows):
        print(line)


def build_ascii_table(
    row_header: str,
    column_headers: tuple[str, ...],
    rows: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[str, ...]:
    table_rows = [(row_header, *column_headers)] + [
        (row_name, *values) for row_name, values in rows
    ]
    widths = [
        max(len(str(row[column_index])) for row in table_rows)
        for column_index in range(len(table_rows[0]))
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: tuple[str, ...]) -> str:
        cells = []
        for column_index, cell in enumerate(row):
            if column_index == 0:
                cells.append(f" {cell:<{widths[column_index]}} ")
            else:
                cells.append(f" {cell:>{widths[column_index]}} ")
        return "|" + "|".join(cells) + "|"

    formatted_rows = [border, format_row(table_rows[0]), border]
    formatted_rows.extend(format_row(row) for row in table_rows[1:])
    formatted_rows.append(border)
    return tuple(formatted_rows)


def format_rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator:.1%}"


def format_average(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00"
    return f"{numerator / denominator:.2f}"


def format_enabled_combo_rules(config: SimulationConfig) -> str:
    if not config.enabled_combo_rules:
        return "none"
    return ", ".join(config.enabled_combo_rules)


if __name__ == "__main__":
    main()
