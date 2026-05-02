"""
Dataset preparation and validation script.

Walks the expected folder structure for the RWF-2000 dataset, verifies every
video file can be opened by OpenCV, and prints a per-split / per-class summary.
Run this before training to catch missing or corrupted files early.

Expected layout:
    datasets/rwf2000/
      train/violence/        <- *.mp4 / *.avi clips
      train/no_violence/
      val/violence/
      val/no_violence/
      test/violence/
      test/no_violence/

Usage:
    cd backend/training
    python scripts/prepare_video_dataset.py --data_root datasets/rwf2000
"""

import argparse
import logging
from pathlib import Path

import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SPLITS      = ["train", "val", "test"]
CLASS_NAMES = ["violence", "no_violence"]
EXTENSIONS  = {".mp4", ".avi", ".mov", ".mkv"}


def check_video(path: Path) -> tuple[bool, str]:
    """Return (ok, message).  Tries to open the file and read one frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, "cannot open"
    ret, _ = cap.read()
    cap.release()
    if not ret:
        return False, "no frames readable"
    return True, "ok"


def validate_split(split_root: Path, verbose: bool) -> dict:
    """
    Validate one split directory.

    Returns a dict:  {class_name: {"total": int, "ok": int, "bad": list[str]}}
    """
    results = {}
    for class_name in CLASS_NAMES:
        class_dir = split_root / class_name
        if not class_dir.is_dir():
            logger.warning("Missing class directory: %s", class_dir)
            results[class_name] = {"total": 0, "ok": 0, "bad": []}
            continue

        total, ok_count, bad = 0, 0, []
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() not in EXTENSIONS:
                continue
            total += 1
            ok, msg = check_video(f)
            if ok:
                ok_count += 1
            else:
                bad.append(f"{f.name}: {msg}")
                if verbose:
                    logger.warning("  BAD  %s — %s", f, msg)

        results[class_name] = {"total": total, "ok": ok_count, "bad": bad}

    return results


def print_summary(data_root: Path, all_results: dict) -> None:
    col = 14
    header = f"  {'split':>{col}}  {'class':>{col}}  {'total':>{col}}  {'ok':>{col}}  {'bad':>{col}}"
    print("\n=== Dataset Summary ===")
    print(header)
    print("  " + "-" * (col * 5 + 10))

    total_clips, total_ok, total_bad = 0, 0, 0
    for split, class_results in all_results.items():
        for class_name, stats in class_results.items():
            n_bad = len(stats["bad"])
            print(
                f"  {split:>{col}}  {class_name:>{col}}"
                f"  {stats['total']:>{col}}  {stats['ok']:>{col}}  {n_bad:>{col}}"
            )
            total_clips += stats["total"]
            total_ok    += stats["ok"]
            total_bad   += n_bad

    print("  " + "-" * (col * 5 + 10))
    print(f"  {'TOTAL':>{col}}  {'':>{col}}  {total_clips:>{col}}  {total_ok:>{col}}  {total_bad:>{col}}")

    if total_bad > 0:
        print(f"\n  [WARNING]  {total_bad} corrupted / unreadable file(s) found -- remove or replace before training.")
    elif total_clips == 0:
        print(f"\n  [WARNING]  No video files found under {data_root}. Place dataset files and re-run.")
    else:
        print(f"\n  [OK]  All {total_clips} clips verified successfully.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate RWF-2000 dataset structure")
    p.add_argument("--data_root", default="datasets/rwf2000")
    p.add_argument("--verbose",   action="store_true", help="Print path of each bad file")
    return p.parse_args()


def main() -> None:
    args      = parse_args()
    data_root = Path(args.data_root)

    if not data_root.is_dir():
        logger.error("Dataset root not found: %s", data_root)
        return

    logger.info("Scanning %s ...", data_root)
    all_results = {}
    for split in SPLITS:
        split_root = data_root / split
        if not split_root.is_dir():
            logger.warning("Split directory missing: %s", split_root)
            all_results[split] = {}
            continue
        logger.info("  Checking split: %s", split)
        all_results[split] = validate_split(split_root, verbose=args.verbose)

    print_summary(data_root, all_results)


if __name__ == "__main__":
    main()
