from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2

from src.kyc_demo.video import VideoInputError, sample_face_frames


def normalize_label(value: str) -> str:
    label = value.strip().lower()
    if label in {"real", "0", "original", "authentic"}:
        return "real"
    if label in {"fake", "1", "manipulated", "synthetic", "forged"}:
        return "fake"
    raise ValueError(f"Unknown label {value!r}; expected real/fake or 0/1.")


def read_manifest(path: Path, root: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"path", "label", "group_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Manifest CSV needs columns: path,label,group_id")
        for row in reader:
            video = Path(row["path"].strip())
            if not video.is_absolute():
                video = root / video
            rows.append({
                "path": video,
                "label": normalize_label(row["label"]),
                "group_id": row["group_id"].strip(),
            })
    if not rows:
        raise ValueError("Manifest contains no video rows.")
    if any(not row["group_id"] for row in rows):
        raise ValueError("Every row needs a non-empty group_id.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create train/val face-crop folders from a labeled video manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Base directory for relative video paths in the CSV.")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-groups", type=int, default=0,
                        help="Optional random subset of groups; 0 means all groups.")
    parser.add_argument("--frames-per-video", type=int, default=4)
    args = parser.parse_args()

    if not 0.05 <= args.val_ratio <= 0.5:
        parser.error("--val-ratio must be between 0.05 and 0.5")
    rows = read_manifest(args.manifest, args.root)
    groups = sorted({row["group_id"] for row in rows})
    if args.max_groups:
        if args.max_groups < 2:
            parser.error("--max-groups must be at least 2")
        random.Random(args.seed).shuffle(groups)
        groups = groups[:args.max_groups]
    if len(groups) < 2:
        parser.error("Need at least two source groups to create train and validation splits.")

    random.Random(args.seed).shuffle(groups)
    val_count = max(1, round(len(groups) * args.val_ratio))
    val_groups = set(groups[:val_count])
    selected_group_set = set(groups)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["group_id"] in selected_group_set:
            by_group[row["group_id"]].append(row)

    # FF++ has several fake methods for many targets. Use one real and one
    # randomly selected fake per target group for this balanced baseline.
    chooser = random.Random(args.seed)
    selected = []
    unmatched_groups = 0
    for group_id in groups:
        members = by_group[group_id]
        real_rows = [row for row in members if row["label"] == "real"]
        fake_rows = [row for row in members if row["label"] == "fake"]
        if not real_rows or not fake_rows:
            unmatched_groups += 1
            continue
        selected.append(chooser.choice(real_rows))
        selected.extend(fake_rows)
    if not selected:
        parser.error("No source groups contained both a real video and a fake video.")

    counts = defaultdict(int)
    skipped = 0
    for row_index, row in enumerate(selected):
        split = "val" if row["group_id"] in val_groups else "train"
        destination = args.output / split / row["label"]
        destination.mkdir(parents=True, exist_ok=True)
        try:
            faces, _ = sample_face_frames(
                row["path"], sample_count=args.frames_per_video
            )
        except (VideoInputError, OSError) as exc:
            print(f"SKIP {row['path']}: {exc}")
            skipped += 1
            continue

        stable_id = hashlib.sha1(
            f"{row['group_id']}|{row['path']}".encode("utf-8")
        ).hexdigest()[:16]
        saved = 0
        for frame_index, face in enumerate(faces):
            output_path = destination / f"{stable_id}_{frame_index:02d}.jpg"
            if cv2.imwrite(str(output_path), face.face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                saved += 1
        counts[(split, row["label"])] += saved
        if row_index % 25 == 0:
            print(f"Processed {row_index + 1}/{len(selected)} videos...")

    print(f"Finished. Skipped videos: {skipped}")
    print(f"Unpaired source groups omitted: {unmatched_groups}")
    for split in ("train", "val"):
        for label in ("fake", "real"):
            print(f"{split}/{label}: {counts[(split, label)]} face crops")
    print(f"Output: {args.output.resolve()}")
    print("Review the counts and sample crops before training.")


if __name__ == "__main__":
    main()
