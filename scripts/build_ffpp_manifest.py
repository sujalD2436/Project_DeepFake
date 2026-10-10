from __future__ import annotations

import argparse
import csv
from pathlib import Path, PurePosixPath


# Start with the paired core FF++ collections. DeepFakeDetection has a
# separate source-video collection and naming convention, so it is excluded.
COLLECTIONS = {
    "original.csv": ("real", "original"),
    "Deepfakes.csv": ("fake", "Deepfakes"),
    "Face2Face.csv": ("fake", "Face2Face"),
    "FaceShifter.csv": ("fake", "FaceShifter"),
    "FaceSwap.csv": ("fake", "FaceSwap"),
    "NeuralTextures.csv": ("fake", "NeuralTextures"),
}


def source_group(relative_path: str, label: str) -> str:
    stem = PurePosixPath(relative_path.replace("\\", "/")).stem
    if label == "real":
        return stem
    target_id, separator, _source_id = stem.partition("_")
    if not separator or not target_id:
        raise ValueError(f"Expected FF++ target_source filename, got {relative_path!r}")
    return target_id


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a path/label/source manifest from FF++ C23 metadata."
    )
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Folder containing original/, manipulation folders, and csv/.")
    parser.add_argument("--output", type=Path, default=Path("raw_data/videos.csv"))
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    csv_root = root / "csv"
    rows: dict[tuple[str, str], dict[str, str]] = {}
    missing = 0

    for csv_name, (expected_label, expected_folder) in COLLECTIONS.items():
        csv_path = csv_root / csv_name
        if not csv_path.is_file():
            raise FileNotFoundError(f"Required file not found: {csv_path}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if not reader.fieldnames or not {"File Path", "Label"}.issubset(reader.fieldnames):
                raise ValueError(
                    f"{csv_name} needs 'File Path' and 'Label' columns; "
                    f"found {reader.fieldnames}"
                )
            for line_number, record in enumerate(reader, start=2):
                relative = (record.get("File Path") or "").strip().replace("\\", "/")
                label = (record.get("Label") or "").strip().lower()
                if not relative:
                    continue
                if label != expected_label:
                    raise ValueError(
                        f"Unexpected label {record.get('Label')!r} in "
                        f"{csv_name}:{line_number}; expected {expected_label.upper()}"
                    )
                parts = PurePosixPath(relative).parts
                if not parts or parts[0].lower() != expected_folder.lower():
                    raise ValueError(
                        f"Unexpected path {relative!r} in {csv_name}; "
                        f"expected it under {expected_folder}/"
                    )
                if not root.joinpath(*parts).is_file():
                    missing += 1
                    continue
                group_id = source_group(relative, label)
                rows[(relative, label)] = {
                    "path": relative,
                    "label": label,
                    "group_id": group_id,
                }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=["path", "label", "group_id"])
        writer.writeheader()
        writer.writerows(rows.values())

    real_count = sum(row["label"] == "real" for row in rows.values())
    fake_count = sum(row["label"] == "fake" for row in rows.values())
    print(f"Manifest: {args.output.resolve()}")
    print(f"Videos indexed: real={real_count}, fake={fake_count}")
    print(f"Missing video files skipped: {missing}")
    print("DeepFakeDetection and summary/shuffled CSVs were excluded.")


if __name__ == "__main__":
    main()

