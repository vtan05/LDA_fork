#!/usr/bin/env python3
from pathlib import Path
import shutil
import re

# ========== EDIT THESE ==========
BVH_DIR        = Path(r"/host_data/van/DTM/results/finedance")         # where your .bvh files are
WAV_DIR        = Path(r"/host_data/van/DTM/results/finedance")         # where your .wav files are
QUARANTINE_DIR = Path(r"/host_data/van/DTM/results/finedance/filter_bvh")   # where to move unmatched .bvh files
RECURSIVE      = False                           # search subfolders
PRESERVE_TREE  = True                           # preserve BVH_DIR subfolders in QUARANTINE_DIR
DRY_RUN        = False                          # True = don't actually move files
# =================================

def collect_wav_stems(wav_root: Path, recursive: bool) -> set:
    """Collect lowercase stems of all .wav files (case-insensitive)."""
    it = wav_root.rglob("*") if recursive else wav_root.glob("*")
    stems = set()
    for p in it:
        if p.is_file() and p.suffix.lower() == ".wav":
            stems.add(p.stem.lower())
    return stems

def canonical_bvh_stem(bvh_path: Path) -> str:
    """
    Remove the trailing style token from BVH stem:
    'finedance_..._0_ClassicHanTang.bvh' -> 'finedance_..._0'
    Rule: strip the last '_[A-Za-z]+' segment if present.
    """
    stem = bvh_path.stem
    m = re.match(r"^(.*)_[A-Za-z]+$", stem)
    return (m.group(1) if m else stem).lower()

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        try:
            src.rename(dst)  # faster if same filesystem
            return
        except Exception:
            pass
    # Handle cross-device or collision
    base, ext = dst.stem, dst.suffix
    k = 1
    while dst.exists():
        dst = dst.with_name(f"{base}__dup{k}{ext}")
        k += 1
    shutil.move(str(src), str(dst))

def main():
    if not BVH_DIR.is_dir(): raise SystemExit(f"[ERR] BVH_DIR not found: {BVH_DIR}")
    if not WAV_DIR.is_dir(): raise SystemExit(f"[ERR] WAV_DIR not found: {WAV_DIR}")
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    wav_stems = collect_wav_stems(WAV_DIR, RECURSIVE)
    pattern_iter = BVH_DIR.rglob("*.bvh") if RECURSIVE else BVH_DIR.glob("*.bvh")

    checked = moved = 0
    for bvh in pattern_iter:
        if not bvh.is_file(): continue
        checked += 1
        canon = canonical_bvh_stem(bvh)
        if canon not in wav_stems:
            if PRESERVE_TREE:
                rel = bvh.parent.relative_to(BVH_DIR)
                dst = QUARANTINE_DIR / rel / bvh.name
            else:
                dst = QUARANTINE_DIR / bvh.name

            if DRY_RUN:
                print(f"[DRY] MOVE {bvh} -> {dst}")
            else:
                try:
                    safe_move(bvh, dst)
                    moved += 1
                    print(f"[MOVE] {bvh} -> {dst}")
                except Exception as e:
                    print(f"[WARN] Failed to move {bvh}: {e}")

    print(f"[DONE] Checked {checked} BVH files; moved {moved} without matching WAV.")

if __name__ == "__main__":
    main()
