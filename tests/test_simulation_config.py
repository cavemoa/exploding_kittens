from argparse import Namespace
from collections import Counter

from scripts.run_simulation import (
    AggregateAgentStats,
    SimulationConfig,
    build_config,
    play_game,
    print_agent_diagnostics,
)


def test_build_config_uses_yaml_values(tmp_path) -> None:
    config_path = tmp_path / "game_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "games: 25",
                "players: 3",
                "strategy: skip-if-possible",
                "seed: 99",
                "max_turns: 750",
                "verbose: true",
                "include_cards:",
                "  - normal",
                "  - skip",
                "exclude_cards:",
                "  - attack",
                "enabled_combo_rules:",
                "  - two_of_a_kind",
            ]
        ),
        encoding="utf-8",
    )

    config = build_config(
        Namespace(
            config=config_path,
            games=None,
            players=None,
            strategy=None,
            seed=None,
            max_turns=None,
            verbose=None,
        )
    )

    assert config == SimulationConfig(
        games=25,
        players=3,
        strategy="skip-if-possible",
        seed=99,
        max_turns=750,
        verbose=True,
        include_cards=("normal", "skip"),
        exclude_cards=("attack",),
        enabled_combo_rules=("two_of_a_kind",),
    )


def test_command_line_values_override_yaml_values(tmp_path) -> None:
    config_path = tmp_path / "game_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "games: 25",
                "players: 3",
                "strategy: random",
                "seed: 99",
                "max_turns: 750",
                "verbose: false",
                "include_cards:",
                "  - normal",
                "  - skip",
                "exclude_cards:",
                "  - attack",
                "enabled_combo_rules: []",
            ]
        ),
        encoding="utf-8",
    )

    config = build_config(
        Namespace(
            config=config_path,
            games=10,
            players=4,
            strategy="draw-only",
            seed=None,
            max_turns=None,
            verbose=True,
            include_cards=("normal", "attack"),
            exclude_cards=(),
            enabled_combo_rules=("two_of_a_kind",),
        )
    )

    assert config == SimulationConfig(
        games=10,
        players=4,
        strategy="draw-only",
        seed=99,
        max_turns=750,
        verbose=True,
        include_cards=("normal", "attack"),
        exclude_cards=(),
        enabled_combo_rules=("two_of_a_kind",),
    )


def test_play_game_returns_diagnostic_counts() -> None:
    result = play_game(
        player_count=2,
        strategy="draw-only",
        seed=1,
        max_turns=200,
        verbose=False,
        include_cards=("normal",),
        exclude_cards=(),
        enabled_combo_rules=(),
    )

    assert result.winner in {"player_1", "player_2"}
    assert result.turns > 0
    assert result.event_counts["draw"] > 0
    assert result.event_counts["eliminated"] == 1
    assert set(result.agent_stats) == {"player_1", "player_2"}
    assert sum(stats.explosions for stats in result.agent_stats.values()) == 1
    assert all(stats.turns_survived > 0 for stats in result.agent_stats.values())
    assert all(stats.hand_size_samples for stats in result.agent_stats.values())


def test_agent_diagnostics_print_as_tables(capsys) -> None:
    agent_totals = {
        "player_1": AggregateAgentStats(
            games=2,
            wins=1,
            turns_survived=30,
            cards_played=Counter({"skip": 2}),
            combos_played=Counter({"two_of_a_kind": 1}),
            explosions=1,
            defuses=2,
            hand_size_total=12,
            hand_size_samples=6,
        ),
        "player_2": AggregateAgentStats(
            games=2,
            wins=1,
            turns_survived=24,
            cards_played=Counter({"skip": 1}),
            explosions=1,
            defuses=0,
            hand_size_total=18,
            hand_size_samples=6,
        ),
    }

    print_agent_diagnostics(agent_totals)

    output = capsys.readouterr().out
    assert "Per-agent Diagnostics" in output
    assert "Summary" in output
    assert "Cards Played / Game" in output
    assert "Combos Played / Game" in output
    assert "| Metric" in output
    assert "| Card" in output
    assert "| Combo" in output
    assert "| Win rate" in output
    assert "player_1: win rate" not in output
