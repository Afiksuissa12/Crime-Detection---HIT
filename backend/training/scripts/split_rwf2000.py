"""
Split the raw RWF-2000 download into train / val / test splits.

Input structure (as downloaded):
    <input_dir>/
      Fight/
      NonFight/

Output structure:
    <output_dir>/
      train/violence/
      train/no_violence/
      val/violence/
      val/no_violence/
      test/violence/
      test/no_violence/

Split ratio  : 70% train / 15% val / 15% test
Split level  : per video — no video appears in more than one split
Shuffle seed : 42 (reproducible)
File action  : copy (original files are never moved or deleted)

Usage:
    cd backend/training
    python scripts/split_rwf2000.py \\
        --input_dir  "C:/path/to/RWF-2000" \\
        --output_dir datasets/rwf2000
"""

import argparse
import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

SEED       = 42
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15   # test gets the remainder (1 - 0.70 - 0.15 = 0.15)

SPLITS     = ["train", "val", "test"]
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Raw folder name → target class name
CLASS_MAP = {
    "Fight":    "violence",
    "NonFight": "no_violence",
}

# Some RWF-2000 downloads are already split into train/val sub-folders.
# The script handles both layouts automatically:
#
#   Flat layout:          <input>/Fight/       <input>/NonFight/
#   Pre-split layout:     <input>/train/Fight/ <input>/train/NonFight/
#                         <input>/val/Fight/   <input>/val/NonFight/
#
# All videos are pooled, then re-split from scratch at 70/15/15.


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def collect_videos(input_root: Path, raw_class_name: str) -> list[Path]:
    """
    Collect all video files for one class, regardless of layout.

    Searches for every folder named raw_class_name anywhere under input_root
    (handles both flat and pre-split layouts) and returns a deduplicated,
    sorted list of video paths.
    """
    seen:   set[str]   = set()
    videos: list[Path] = []

    for folder in input_root.rglob(raw_class_name):
        if not folder.is_dir():
            continue
        for f in sorted(folder.iterdir()):
            if f.is_file() and f.suffix.lower() in EXTENSIONS and f.name not in seen:
                seen.add(f.name)
                videos.append(f)

    videos.sort(key=lambda p: p.name)
    if not videos:
        logger.warning("No video files found for class '%s' under %s", raw_class_name, input_root)
    return videos


def split_videos(videos: list[Path]) -> dict[str, list[Path]]:
    """
    Shuffle and split a list of video paths into train / val / test.

    Uses seed 42 for reproducibility.  Each video appears in exactly one split.
    """
    videos = list(videos)   # copy so we don't mutate caller's list
    random.seed(SEED)
    random.shuffle(videos)

    n       = len(videos)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)
    # test gets whatever is left to avoid rounding loss
    n_test  = n - n_train - n_val

    return {
        "train": videos[:n_train],
        "val":   videos[n_train : n_train + n_val],
        "test":  videos[n_train + n_val :],
    }


def copy_split(
    split_map:   dict[str, list[Path]],
    class_name:  str,
    output_root: Path,
) -> dict[str, int]:
    """
    Copy files from split_map into output_root/<split>/<class_name>/.

    Returns a dict {split_name: files_copied}.
    """
    counts: dict[str, int] = {}
    for split_name, files in split_map.items():
        dest_dir = output_root / split_name / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        for src in files:
            dest = dest_dir / src.name
            if dest.exists():
                logger.debug("Skip (already exists): %s", dest)
            else:
                shutil.copy2(src, dest)

        counts[split_name] = len(files)
        logger.info(
            "  %s / %-12s  %d files  ->  %s",
            split_name, class_name, len(files), dest_dir,
        )

    return counts


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(all_counts: dict[str, dict[str, int]], total_per_class: dict[str, int]) -> None:
    col = 14
    header = f"  {'split':>{col}}  {'class':>{col}}  {'count':>{col}}"
    print("\n=== Split Summary ===")
    print(header)
    print("  " + "-" * (col * 3 + 6))

    grand_total = 0
    for split in SPLITS:
        for class_name in CLASS_MAP.values():
            count = all_counts.get(split, {}).get(class_name, 0)
            print(f"  {split:>{col}}  {class_name:>{col}}  {count:>{col}}")
            grand_total += count

    print("  " + "-" * (col * 3 + 6))
    for class_name, total in total_per_class.items():
        print(f"  {'total':>{col}}  {class_name:>{col}}  {total:>{col}}")
    print(f"  {'GRAND TOTAL':>{col}}  {'':>{col}}  {grand_total:>{col}}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split RWF-2000 into train/val/test")
    p.add_argument(
        "--input_dir",
        required=True,
        help='Path to the raw RWF-2000 folder containing Fight/ and NonFight/',
    )
    p.add_argument(
        "--output_dir",
        default="datasets/rwf2000",
        help="Destination root (default: datasets/rwf2000)",
    )
    return p.parse_args()


def main() -> None:
    args        = parse_args()
    input_root  = Path(args.input_dir)
    output_root = Path(args.output_dir)

    # Validate input
    if not input_root.is_dir():
        logger.error("Input directory not found: %s", input_root)
        return

    logger.info("Input  : %s", input_root)
    logger.info("Output : %s", output_root)
    logger.info("Split  : %.0f%% train / %.0f%% val / %.0f%% test  (seed=%d)",
                TRAIN_FRAC * 100, VAL_FRAC * 100, (1 - TRAIN_FRAC - VAL_FRAC) * 100, SEED)

    all_counts:      dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    total_per_class: dict[str, int]            = {}

    for raw_name, class_name in CLASS_MAP.items():
        videos = collect_videos(input_root, raw_name)
        logger.info("Found %d videos for class '%s'", len(videos), raw_name)

        split_map              = split_videos(videos)
        total_per_class[class_name] = len(videos)

        counts = copy_split(split_map, class_name, output_root)
        for split_name, count in counts.items():
            all_counts[split_name][class_name] = count

    print_summary(all_counts, total_per_class)
    logger.info("Done. Dataset ready at: %s", output_root.resolve())


if __name__ == "__main__":
    main()
