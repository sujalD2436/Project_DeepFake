from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.kyc_demo.config import MODEL_PATH
from src.kyc_demo.scoring import VideoScorer


def main():
    parser = argparse.ArgumentParser(description="Score a short video with the trained model.")
    parser.add_argument("video", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    scorer = VideoScorer(args.checkpoint)
    print(json.dumps(scorer.score(args.video), indent=2))


if __name__ == "__main__":
    main()

