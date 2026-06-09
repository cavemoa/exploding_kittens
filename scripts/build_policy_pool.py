from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exploding_kittens.policy_pool import (  # noqa: E402
    POLICY_POOL_SAMPLE_MODES,
    PolicyPool,
)


def build_policy_pool(
    *,
    checkpoint_dir: Path,
    metadata_path: Path,
    players: int,
    sample_mode: str,
    seed: int,
    scores: dict[str, float],
) -> dict[str, str]:
    pool = PolicyPool.from_checkpoint_dir(checkpoint_dir, scores=scores)
    pool.save_metadata(metadata_path)
    return pool.sample_opponents(
        players=players,
        mode=sample_mode,
        rng=random.Random(seed),
    )


def load_scores(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scores file must contain a JSON object.")
    return {str(key): float(value) for key, value in payload.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build self-play policy-pool metadata and preview sampling."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("reports/policy_pool.json"),
    )
    parser.add_argument("--players", type=int, default=3)
    parser.add_argument("--sample-mode", choices=POLICY_POOL_SAMPLE_MODES, default="mixed")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scores", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sampled = build_policy_pool(
        checkpoint_dir=args.checkpoint_dir,
        metadata_path=args.metadata_path,
        players=args.players,
        sample_mode=args.sample_mode,
        seed=args.seed,
        scores=load_scores(args.scores),
    )

    print("Policy Pool")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Metadata: {args.metadata_path}")
    print(f"Sample mode: {args.sample_mode}")
    print(json.dumps(sampled, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
