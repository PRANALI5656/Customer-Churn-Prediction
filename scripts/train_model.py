from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model import DEFAULT_ARTIFACT_DIR, DEFAULT_DATA_PATH, train_and_save


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save the churn prediction model.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    metadata = train_and_save(args.data, args.artifacts)
    print(f"Saved model artifacts to {args.artifacts.resolve()}")
    print(json.dumps(metadata["holdout_metrics"], indent=2))


if __name__ == "__main__":
    main()
