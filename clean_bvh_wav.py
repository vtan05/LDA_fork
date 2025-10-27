#!/usr/bin/env python3
"""
clean_bvh_wav.py

- Renames BVH files that start with "test_" by removing the prefix.
- Deletes WAV files that have no equivalent BVH file (matching filename stem).
  * Matching is done on a *normalized* stem where any trailing "_slice<digits>"
    is removed for comparison (so "foo_slice0" matches "foo").

USAGE
-----
python clean_bvh_wav.py --bvh-dir /host_data/van/edge_eval_34s --wav-dir /host_data/van/edge_eval_34s --commit

Defaults: both --bvh-dir and --wav-dir default to the current directory.

Tips:
- Omit --commit to do a dry run (no changes). Add --commit to actually rename/delete.
"""

from pathlib import Path
import re
import argparse
import sys

SLICE_RE = re.compile(r"_slice\d+$", re.IGNORECASE)

def normalize_stem(stem: str) -> str:
    """Remove a trailing '_slice<digits>' from a filename stem for comparison."""
    return SLICE_RE.sub("", stem)

def remove_test_prefix_and_collect_stems(bvh_dir: Path, commit: bool) -> set[str]:
    """
    Find BVH files, remove leading 'test_' if present, and collect normalized stems.
    Returns a set of normalized stems after (prospective) renaming.
    """
    stems = set()
    for bvh_path in bvh_dir.rglob("*.bvh"):
        old_name = bvh_path.name
        parent = bvh_path.parent

        # If it starts with "test_", plan to rename
        if old_name.startswith("test_"):
            new_name = old_name[len("test_"):]
            new_path = parent / new_name

            if new_path.exists():
                # Avoid overwriting collisions
                print(f"[WARN] Target exists, skipping rename:\n       {bvh_path} -> {new_path}")
                final_path = bvh_path  # Keep old for stem collection
            else:
                print(f"[RENAME]{' (dry-run)' if not commit else ''} {bvh_path} -> {new_path}")
                final_path = new_path
                if commit:
                    try:
                        bvh_path.rename(new_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to rename {bvh_path} -> {new_path}: {e}", file=sys.stderr)
                        final_path = bvh_path  # fall back
        else:
            final_path = bvh_path

        # Collect normalized stem
        stem = final_path.stem
        norm = normalize_stem(stem)
        stems.add(norm)

    return stems

def delete_orphan_wavs(wav_dir: Path, valid_bvh_stems: set[str], commit: bool) -> None:
    """
    Delete WAV files whose (normalized) stem does not exist among BVH stems.
    """
    for wav_path in wav_dir.rglob("*.wav"):
        wav_stem_norm = normalize_stem(wav_path.stem)
        if wav_stem_norm not in valid_bvh_stems:
            print(f"[DELETE]{' (dry-run)' if not commit else ''} {wav_path} (no matching BVH)")
            if commit:
                try:
                    wav_path.unlink()
                except Exception as e:
                    print(f"[ERROR] Failed to delete {wav_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Rename BVHs (drop 'test_') and delete WAVs without matching BVHs.")
    parser.add_argument("--bvh-dir", type=Path, default=Path.cwd(), help="Directory containing .bvh files (recursively).")
    parser.add_argument("--wav-dir", type=Path, default=Path.cwd(), help="Directory containing .wav files (recursively).")
    parser.add_argument("--commit", action="store_true", help="Apply changes. Without this flag, runs as a dry-run.")
    args = parser.parse_args()

    if not args.bvh_dir.exists():
        print(f"[ERROR] BVH directory does not exist: {args.bvh_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.wav_dir.exists():
        print(f"[ERROR] WAV directory does not exist: {args.wav_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"BVH dir: {args.bvh_dir}")
    print(f"WAV dir: {args.wav_dir}")
    print(f"Mode: {'COMMIT' if args.commit else 'DRY-RUN'}")

    bvh_stems = remove_test_prefix_and_collect_stems(args.bvh_dir, commit=args.commit)

    print(f"[INFO] Collected {len(bvh_stems)} BVH stems (normalized).")
    delete_orphan_wavs(args.wav_dir, bvh_stems, commit=args.commit)

if __name__ == "__main__":
    main()
