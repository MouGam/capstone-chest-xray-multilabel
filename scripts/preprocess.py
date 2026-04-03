"""
NIH ChestX-ray14 전처리 파이프라인
Step 1: 31장 수동 이미지 CLAHE 적용
Step 2: 유효/제거 이미지 목록 생성
Step 3: CLAHE + Resize 224x224 + 3채널 → nih-dataset/resized/images/
Step 5: available/unavailable 디렉토리 구성 + data.csv
Step 6: Multi-hot Encoding
"""

import os
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool, cpu_count
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "nih-dataset" / "raw"
BY_HAND_FINAL = PROJECT_ROOT / "by_hand" / "final"
BY_HAND_CLAHE = PROJECT_ROOT / "by_hand" / "clahe"
RESIZED_DIR = PROJECT_ROOT / "nih-dataset" / "resized" / "images"
PROCESSED_DIR = PROJECT_ROOT / "nih-dataset" / "processed"
STATS_CSV = PROJECT_ROOT / "quality_filter_result" / "all_image_stats.csv"

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# 수동 처리 대상 31장
MANUAL_DARK = [
    "00003465_002.png", "00003465_006.png", "00009621_004.png", "00009621_005.png",
    "00010805_011.png", "00011553_008.png", "00012654_001.png", "00012742_000.png",
    "00014982_000.png", "00015007_005.png", "00015007_006.png", "00015462_000.png",
    "00015462_001.png", "00016292_003.png", "00018251_004.png", "00018251_008.png",
    "00018251_012.png", "00019534_000.png", "00019895_001.png", "00019967_021.png",
    "00022339_000.png", "00022723_000.png", "00022815_012.png", "00027765_000.png",
    "00028474_000.png", "00030320_006.png", "00030609_019.png", "00030609_020.png",
]
MANUAL_WHITE = ["00005618_000.png", "00006094_000.png"]
MANUAL_STD = ["00004480_000.png"]
MANUAL_ALL = set(MANUAL_DARK + MANUAL_WHITE + MANUAL_STD)

# 14개 질환 알파벳 순
DISEASES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax",
]


def build_lookup() -> dict[str, str]:
    """raw 디렉토리 내 파일명 → 전체 경로 lookup"""
    lookup = {}
    for p in glob.glob(str(RAW_DIR / "images" / "*.png")):
        lookup[os.path.basename(p)] = p
    for i in range(1, 12):
        for p in glob.glob(str(RAW_DIR / f"images_{i:03d}" / "images" / "*.png")):
            lookup[os.path.basename(p)] = p
    return lookup


def step1_clahe_manual():
    """Step 1: by_hand/final/ 31장에 CLAHE 적용 → by_hand/clahe/"""
    print("=== Step 1: 31장 CLAHE 적용 ===")
    BY_HAND_CLAHE.mkdir(parents=True, exist_ok=True)

    for f in sorted(os.listdir(BY_HAND_FINAL)):
        if not f.endswith(".png"):
            continue
        img = cv2.imread(str(BY_HAND_FINAL / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: {f} 읽기 실패")
            continue
        result = CLAHE.apply(img)
        cv2.imwrite(str(BY_HAND_CLAHE / f), result)

    print(f"  완료: {len(os.listdir(BY_HAND_CLAHE))}장")


def step2_build_lists(stats: pd.DataFrame) -> tuple[set[str], set[str]]:
    """Step 2: available/unavailable 목록 생성 (mean/std 기반 사전 필터링)"""
    print("\n=== Step 2: 유효/제거 이미지 목록 생성 ===")

    # 사전 제거 대상: mean < 50 (수동 28장 제외), mean > 195 (수동 2장 제외)
    pre_remove = set()

    dark = stats[stats["mean"] < 50]
    for _, row in dark.iterrows():
        if row["filename"] not in MANUAL_ALL:
            pre_remove.add(row["filename"])

    bright = stats[stats["mean"] > 195]
    for _, row in bright.iterrows():
        if row["filename"] not in MANUAL_ALL:
            pre_remove.add(row["filename"])

    print(f"  사전 제거 (mean 기준): {len(pre_remove)}장")

    # CLAHE 후 std < 25 체크는 Step 3에서 동적으로 처리
    # 여기서는 사전 제거 목록만 반환
    all_files = set(stats["filename"].tolist())
    pre_available = all_files - pre_remove

    print(f"  사전 available 후보: {len(pre_available)}장")
    return pre_available, pre_remove


def process_single_image(args):
    """단일 이미지 처리: CLAHE → std 체크 → Resize → 3채널 → 저장"""
    filename, src_path, is_manual = args

    if is_manual:
        # 수동 이미지: by_hand/clahe/에서 로드 (이미 CLAHE 적용됨)
        img = cv2.imread(str(BY_HAND_CLAHE / filename), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (filename, "read_error")
    else:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return (filename, "read_error")
        # CLAHE 적용
        img = CLAHE.apply(img)

    # CLAHE 후 std 체크
    if filename not in MANUAL_ALL:
        img_std = float(np.std(img))
        if img_std < 25:
            return (filename, "low_std")

    # Resize 224x224 Bilinear
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 3채널 복제
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 저장
    cv2.imwrite(str(RESIZED_DIR / filename), img_rgb)
    return (filename, "ok")


def step3_clahe_resize(pre_available: set[str], lookup: dict[str, str]) -> tuple[set[str], set[str]]:
    """Step 3: CLAHE + Resize → nih-dataset/resized/images/"""
    print("\n=== Step 3: CLAHE + Resize 224x224 + 3채널 ===")
    RESIZED_DIR.mkdir(parents=True, exist_ok=True)

    # 작업 목록 생성
    tasks = []
    for filename in sorted(pre_available):
        is_manual = filename in MANUAL_ALL
        src_path = lookup.get(filename, "")
        tasks.append((filename, src_path, is_manual))

    # 멀티프로세싱
    workers = min(cpu_count(), 8)
    print(f"  {len(tasks)}장 처리 시작 ({workers} workers)")

    post_remove = set()
    ok_count = 0

    with Pool(workers) as pool:
        for result in tqdm(pool.imap(process_single_image, tasks, chunksize=256),
                           total=len(tasks), desc="  Processing"):
            filename, status = result
            if status == "ok":
                ok_count += 1
            else:
                post_remove.add(filename)

    final_available = pre_available - post_remove
    print(f"  CLAHE 후 추가 제거 (std<25): {len(post_remove)}장")
    print(f"  최종 resized: {ok_count}장")
    return final_available, post_remove


def step5_organize(available: set[str], unavailable: set[str],
                   meta_df: pd.DataFrame, lookup: dict[str, str]):
    """Step 5: available/unavailable 디렉토리 구성"""
    print("\n=== Step 5: available/unavailable 구성 ===")

    for subset_name, subset_files in [("available", available), ("unavailable", unavailable)]:
        subset_dir = PROCESSED_DIR / subset_name / "images"
        subset_dir.mkdir(parents=True, exist_ok=True)

        if subset_name == "available":
            # resized에서 symlink
            for f in tqdm(sorted(subset_files), desc=f"  {subset_name} symlink"):
                src = RESIZED_DIR / f
                dst = subset_dir / f
                if dst.exists():
                    dst.unlink()
                if src.exists():
                    os.symlink(src.resolve(), dst)
        else:
            # unavailable: raw에서 복사 (전처리 안 된 원본)
            for f in tqdm(sorted(subset_files), desc=f"  {subset_name} copy"):
                src = lookup.get(f)
                if src:
                    shutil.copy2(src, subset_dir / f)

    print(f"  available: {len(available)}장")
    print(f"  unavailable: {len(unavailable)}장")


def step6_multihot_encoding(available: set[str], unavailable: set[str],
                            meta_df: pd.DataFrame):
    """Step 6: Multi-hot Encoding + data.csv 생성"""
    print("\n=== Step 6: Multi-hot Encoding ===")

    def encode_labels(finding_labels: str) -> list[int]:
        labels = [l.strip() for l in str(finding_labels).split("|")]
        encoding = [0] * len(DISEASES)
        for label in labels:
            if label in DISEASES:
                idx = DISEASES.index(label)
                encoding[idx] = 1
        return encoding

    for subset_name, subset_files in [("available", available), ("unavailable", unavailable)]:
        subset_df = meta_df[meta_df["Image Index"].isin(subset_files)].copy()

        # multi-hot encoding
        encodings = subset_df["Finding Labels"].apply(encode_labels)
        for i, disease in enumerate(DISEASES):
            subset_df[disease] = encodings.apply(lambda x: x[i])

        out_path = PROCESSED_DIR / subset_name / "data.csv"
        subset_df.to_csv(out_path, index=False)
        print(f"  {subset_name}/data.csv: {len(subset_df)} rows")


def step7_verify(available: set[str], unavailable: set[str]):
    """검증"""
    print("\n=== 검증 ===")

    resized_count = len(list(RESIZED_DIR.glob("*.png")))
    avail_img_count = len(list((PROCESSED_DIR / "available" / "images").glob("*.png")))
    unavail_img_count = len(list((PROCESSED_DIR / "unavailable" / "images").glob("*.png")))

    avail_csv = pd.read_csv(PROCESSED_DIR / "available" / "data.csv")
    unavail_csv = pd.read_csv(PROCESSED_DIR / "unavailable" / "data.csv")

    print(f"  resized/images/: {resized_count}장")
    print(f"  available/images/: {avail_img_count}장")
    print(f"  available/data.csv: {len(avail_csv)} rows")
    print(f"  unavailable/images/: {unavail_img_count}장")
    print(f"  unavailable/data.csv: {len(unavail_csv)} rows")
    print(f"  합계: {avail_img_count + unavail_img_count}장 (expected: 112120)")

    # shape 확인
    sample = cv2.imread(str(next(RESIZED_DIR.glob("*.png"))))
    print(f"  sample shape: {sample.shape} (expected: (224, 224, 3))")

    # multi-hot 검증
    no_finding = avail_csv[avail_csv["Finding Labels"] == "No Finding"]
    if len(no_finding) > 0:
        row = no_finding.iloc[0]
        encoding = [row[d] for d in DISEASES]
        print(f"  No Finding encoding: {encoding} (expected: all zeros)")

    # 31장 수동 이미지 확인
    manual_in_avail = sum(1 for f in MANUAL_ALL if f in available)
    print(f"  수동 31장 중 available: {manual_in_avail}/31")


def main():
    print("=" * 60)
    print("NIH ChestX-ray14 전처리 파이프라인")
    print("=" * 60)

    # 데이터 로드
    stats = pd.read_csv(STATS_CSV)
    meta_df = pd.read_csv(RAW_DIR / "Data_Entry_2017.csv")
    lookup = build_lookup()
    print(f"전체 이미지: {len(stats)}장, 메타데이터: {len(meta_df)} rows")

    # Step 1
    step1_clahe_manual()

    # Step 2
    pre_available, pre_remove = step2_build_lists(stats)

    # Step 3
    final_available, post_remove = step3_clahe_resize(pre_available, lookup)

    # 최종 unavailable 합산
    all_unavailable = pre_remove | post_remove
    print(f"\n최종: available={len(final_available)}, unavailable={len(all_unavailable)}")

    # Step 5
    step5_organize(final_available, all_unavailable, meta_df, lookup)

    # Step 6
    step6_multihot_encoding(final_available, all_unavailable, meta_df)

    # 검증
    step7_verify(final_available, all_unavailable)

    print("\n=== 완료 ===")


if __name__ == "__main__":
    main()
