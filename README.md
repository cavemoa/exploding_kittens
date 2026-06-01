# Exploding Kittens Simulator

A small Python project for learning how to model a card game, test the rules, and later connect simple agents or reinforcement learning environments.

The first milestone is intentionally plain Python. PettingZoo can come later as a wrapper around the engine once the rules are stable.

## What Exists Now

- A simplified game engine in `src/exploding_kittens/engine.py`
- Card classes in `src/exploding_kittens/cards.py`
- Actions represented as an enum in `src/exploding_kittens/actions.py`
- Three simple agents:
  - `RandomAgent`
  - `DrawOnlyAgent`
  - `SkipIfPossibleAgent`
- A simulation script in `scripts/run_simulation.py`
- A focused pytest suite in `tests/test_engine.py`

## Simplified Rules

- The game supports two or more players.
- Each player starts with one `Defuse`.
- On a turn, the current player may draw or play `Skip` if they have one.
- Drawing a normal card adds it to the player's hand.
- Drawing an exploding kitten eliminates the player unless they have a `Defuse`.
- A used `Defuse` is discarded and the exploding kitten goes to the bottom of the draw pile.
- The game ends when one player remains.

## Architecture

The engine coordinates shared game flow: whose turn it is, drawing from the deck, discarding cards, advancing turns, and detecting the winner.

Agents choose concrete `GameAction` objects. A `GameAction` includes the action kind plus any choices needed to perform it, such as hand indexes and target player:

```python
GameAction(
    kind=Action.PLAY_FAVOR,
    card_indexes=(1,),
    target_player="player_3",
)
```

The `Action` enum is now the `GameAction.kind` value. The engine accepts `Action.DRAW` as a small debug convenience, but non-draw moves should always be submitted as full `GameAction` objects so there is no hidden target or card selection.

Agents receive a `GameObservation` instead of direct access to the full engine. The observation includes:

- exact names of the agent's own hand cards, in hand-index order
- own hand counts
- visible discard pile counts
- draw pile size
- number of living players
- current and next-player attack turn debt
- player-specific known top cards from `SeeTheFutureCard`
- public event kinds
- legal concrete actions

This creates a clean boundary between full game state and what an agent is allowed to use for decisions.

Agents can inspect which own-hand cards an action would spend:

```python
for action, card_names in observation.legal_actions_with_card_names():
    ...
```

For example, a Two of a Kind action with `card_indexes=(2, 3)` can be mapped back to `("skip", "skip")` before the agent decides whether that pair is worth spending.

Hidden information stays hidden from observations:

- draw pile order
- exact cards in other players' hands

Opponent card counts are hidden by default. They can be revealed as simple hand sizes by setting:

```yaml
reveal_opponent_card_counts: true
```

Each card class owns its local rule effect:

- `NormalCard.on_draw()` adds the card to the player's hand.
- `SkipCard.play()` discards the skip card and ends the player's turn without drawing.
- `AttackCard.play()` discards the attack card and makes the next player take two turns.
- `ShuffleCard.play()` discards the shuffle card and shuffles the draw pile.
- `FavorCard.play()` targets the next living player with cards and takes one random card from their hand.
- `AlterTheFutureCard.play()` reverses the top three draw-pile cards. This is a custom learning card for this simulator, not a card from the commercial game.
- `SeeTheFutureCard.play()` stores the current top three draw-pile card names in that player's observations until the draw pile changes.
- `ExplodingKittenCard.on_draw()` either eliminates the player or asks a `DefuseCard` to save them.
- `DefuseCard.use_to_defuse()` discards the defuse and puts the exploding kitten back into the draw pile.

This keeps new card rules close to the card class while leaving global game state in the engine.

## Setup

```powershell
venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Run Tests

```powershell
venv\Scripts\python.exe -m pytest
```

## Run A Simulation

The runner automatically reads `game_config.yaml` when it exists:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py
```

Current config options:

```yaml
games: 1000
players: 2
strategy: random
seed: 1
max_turns: 500
verbose: false
reveal_opponent_card_counts: false
include_cards:
exclude_cards: []
enabled_combo_rules: []
```

Command-line arguments override the YAML values:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --games 100 --players 3 --strategy random
```

Optional cards can be selected for focused experiments:

```yaml
include_cards:
  - normal
  - skip
exclude_cards:
  - attack
```

Valid optional card names are `normal`, `skip`, `attack`, `shuffle`, `favor`, `alter_the_future`, and `see_the_future`. `defuse` and `exploding_kitten` are always included because the current game-ending rules depend on them.

`alter_the_future` is a custom learning card for this simulator. It is included so we can experiment with inspecting and reordering the draw pile before implementing more advanced agent choices.

The same card filters can be overridden on the command line with comma-separated values:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --include-cards normal,skip --exclude-cards attack
```

Combo rules are separate from individual card classes. They look for patterns in a player's hand and can be enabled independently:

```yaml
enabled_combo_rules:
  - two_of_a_kind
```

Known combo rules:

- `two_of_a_kind`: play any two cards with the same title, discard both, then steal one random card from the next living player with cards.

Internally, combo rules also generate concrete `GameAction` choices. For example, Two of a Kind specifies which two hand indexes are being played and which player is targeted.

Combo rules can also be enabled from the command line:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --enabled-combo-rules two_of_a_kind
```

The simulation output includes diagnostics that help spot rule bugs:

- finished games vs games that hit `max_turns`
- average turns across all games
- average turns to win among completed games
- shortest and longest game length
- draws, skips, attacks, shuffles, favors, Alter the Future plays, See the Future plays, defuses, and eliminations per game
- combo rule plays such as Two of a Kind
- per-agent ASCII tables for win rate, average turns survived, cards played by type, combos played, explosions, defuses, and average hand size

Try the other baseline strategies:

```powershell
venv\Scripts\python.exe scripts\run_simulation.py --games 1000 --strategy draw-only
venv\Scripts\python.exe scripts\run_simulation.py --games 1000 --strategy skip-if-possible
```

## Good Next Steps

1. Add more deterministic rule tests.
2. Add more card rules one at a time.
3. Add an observation/action-mask layer.
4. Teach simple agents to use `known_top_cards`.
5. Wrap the stable engine in PettingZoo.
