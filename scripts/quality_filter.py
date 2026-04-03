"""
이미지 품질 필터링 스크립트
- 너무 밝거나 어두운 이미지를 탐지
- 일반적 기준: mean intensity < 30 (dark), > 225 (bright), std < 10 (flat)
- 결과를 CSV로 저장하고, 필터링된 이미지 샘플을 별도 디렉토리에 복사
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === Config ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "quality_filter_result"

DARK_THRESHOLD = 30      # mean intensity < 30
BRIGHT_THRESHOLD = 225   # mean intensity > 225
LOW_STD_THRESHOLD = 10   # std < 10 (almost uniform/blank)

def collect_image_paths(raw_dir: Path) -> list[str]:
    """raw 디렉토리 내 모든 이미지 경로 수집 (images, images_001~011)"""
    patterns = [
        str(raw_dir / "images" / "*.png"),
    ]
    for i in range(1, 12):
        patterns.append(str(raw_dir / f"images_{i:03d}" / "*.png"))

    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    return sorted(paths)

def analyze_image(path: str) -> dict:
    """이미지의 mean, std, min, max intensity 계산"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"path": path, "filename": os.path.basename(path),
                "mean": -1, "std": -1, "min": -1, "max": -1, "status": "read_error"}

    mean_val = float(np.mean(img))
    std_val = float(np.std(img))
    min_val = float(np.min(img))
    max_val = float(np.max(img))

    if mean_val < DARK_THRESHOLD:
        status = "too_dark"
    elif mean_val > BRIGHT_THRESHOLD:
        status = "too_bright"
    elif std_val < LOW_STD_THRESHOLD:
        status = "low_contrast"
    else:
        status = "ok"

    return {
        "path": path,
        "filename": os.path.basename(path),
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "min": min_val,
        "max": max_val,
        "status": status,
    }

def save_sample_images(df_filtered: pd.DataFrame, output_dir: Path, max_per_category: int = 20):
    """필터링된 이미지 샘플을 카테고리별로 복사"""
    for status in ["too_dark", "too_bright", "low_contrast", "read_error"]:
        subset = df_filtered[df_filtered["status"] == status]
        if len(subset) == 0:
            continue

        cat_dir = output_dir / status
        cat_dir.mkdir(parents=True, exist_ok=True)

        sample = subset.head(max_per_category)
        for _, row in sample.iterrows():
            src = row["path"]
            dst = cat_dir / row["filename"]
            shutil.copy2(src, dst)

def plot_distribution(df: pd.DataFrame, output_dir: Path):
    """mean intensity 분포 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean intensity distribution
    axes[0].hist(df["mean"], bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(DARK_THRESHOLD, color="blue", linestyle="--", label=f"Dark < {DARK_THRESHOLD}")
    axes[0].axvline(BRIGHT_THRESHOLD, color="red", linestyle="--", label=f"Bright > {BRIGHT_THRESHOLD}")
    axes[0].set_xlabel("Mean Pixel Intensity")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mean Intensity Distribution")
    axes[0].legend()

    # Std distribution
    axes[1].hist(df["std"], bins=100, color="coral", edgecolor="black", alpha=0.7)
    axes[1].axvline(LOW_STD_THRESHOLD, color="blue", linestyle="--", label=f"Low Contrast < {LOW_STD_THRESHOLD}")
    axes[1].set_xlabel("Std of Pixel Intensity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Intensity Std Distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "intensity_distribution.png", dpi=150)
    plt.close()

def main():
    print(f"=== Image Quality Filter ===")
    print(f"Thresholds: dark<{DARK_THRESHOLD}, bright>{BRIGHT_THRESHOLD}, low_std<{LOW_STD_THRESHOLD}")
    print()

    # 이미지 경로 수집
    paths = collect_image_paths(RAW_DIR)
    print(f"Total images found: {len(paths)}")

    # 분석
    results = []
    for p in tqdm(paths, desc="Analyzing images"):
        results.append(analyze_image(p))

    df = pd.DataFrame(results)

    # 결과 요약
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    status_counts = df["status"].value_counts()
    print("\n=== Results ===")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print(f"  Total filtered: {len(df[df['status'] != 'ok'])}")

    # 필터링된 이미지 목록 저장
    df.to_csv(OUTPUT_DIR / "all_image_stats.csv", index=False)
    df_filtered = df[df["status"] != "ok"]
    df_filtered.to_csv(OUTPUT_DIR / "filtered_images.csv", index=False)

    # 샘플 이미지 복사
    save_sample_images(df_filtered, OUTPUT_DIR / "samples")

    # 분포 플롯
    plot_distribution(df, OUTPUT_DIR)

    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"  - all_image_stats.csv: 전체 이미지 통계")
    print(f"  - filtered_images.csv: 필터링 대상 이미지 목록")
    print(f"  - samples/: 카테고리별 샘플 이미지 (최대 20장씩)")
    print(f"  - intensity_distribution.png: 분포 히스토그램")

if __name__ == "__main__":
    main()
