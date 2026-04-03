"""
전체 이미지(112K)에 대한 픽셀 통계(mean, std, min, max) 수집
결과: quality_filter_result/all_image_stats.csv
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "quality_filter_result"


def collect_image_paths(raw_dir: Path) -> list[str]:
    patterns = [str(raw_dir / "images" / "*.png")]
    for i in range(1, 12):
        # images_001/images/*.png 구조
        patterns.append(str(raw_dir / f"images_{i:03d}" / "images" / "*.png"))
        # images_001/*.png (혹시 직접 있는 경우)
        patterns.append(str(raw_dir / f"images_{i:03d}" / "*.png"))
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return sorted(set(paths))


def analyze_image(path: str) -> tuple:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (os.path.basename(path), -1.0, -1.0, -1.0, -1.0)
    return (
        os.path.basename(path),
        round(float(np.mean(img)), 2),
        round(float(np.std(img)), 2),
        float(np.min(img)),
        float(np.max(img)),
    )


def main():
    paths = collect_image_paths(RAW_DIR)
    print(f"Total images: {len(paths)}")

    workers = min(cpu_count(), 8)
    print(f"Using {workers} workers")

    results = []
    with Pool(workers) as pool:
        for r in tqdm(pool.imap(analyze_image, paths, chunksize=256),
                      total=len(paths), desc="Collecting stats"):
            results.append(r)

    df = pd.DataFrame(results, columns=["filename", "mean", "std", "min", "max"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "all_image_stats.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print(f"Total: {len(df)}")
    errors = len(df[df["mean"] < 0])
    if errors:
        print(f"Read errors: {errors}")
    print(f"\nMean intensity — min: {df['mean'].min()}, max: {df['mean'].max()}, "
          f"avg: {df['mean'].mean():.2f}")
    print(f"Std intensity  — min: {df['std'].min()}, max: {df['std'].max()}, "
          f"avg: {df['std'].mean():.2f}")


if __name__ == "__main__":
    main()
