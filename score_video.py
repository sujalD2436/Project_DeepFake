from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.kyc_demo.scoring import VideoScorer


def main():
    parser = argparse.ArgumentParser(description="Score a short video with the trained model.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/deepfake_baseline.pt"))
    args = parser.parse_args()
    scorer = VideoScorer(args.checkpoint)
    print(json.dumps(scorer.score(args.video), indent=2))


if __name__ == "__main__":
    main()

