# NIH ChestX-ray14 Multi-label Classification

흉부 X-ray 이미지에서 14개 질환을 동시 분류하는 멀티라벨 분류 시스템.

## 프로젝트 구조

```
proj/
├── scripts/
│   ├── chestxray_train.py      # 학습 스크립트 (DenseNet-121, Focal Loss, 5-Fold CV)
│   ├── run_training.sh         # gamma 0,1,2 자동 실행 + 웹훅 알림
│   ├── preprocess.py           # 데이터 전처리 파이프라인
│   ├── collect_image_stats.py  # 이미지 통계 수집
│   └── quality_filter.py       # 품질 필터링
├── docs/
│   └── training_guide.md       # 트레이닝 환경 설정 및 실행 가이드
├── plan/
│   ├── 기능요구사항및개발지침.md
│   └── raw/
├── Dockerfile.train
├── requirements-train.txt
└── CLAUDE.md
```

## 기술 스택

| 항목 | 스펙 |
|------|------|
| 모델 | DenseNet-121 (ImageNet Pretrained) |
| 손실함수 | Focal Loss (gamma=0,1,2 실험) |
| 옵티마이저 | AdamW + Cosine Annealing |
| 교차검증 | 5-Fold GroupKFold (Patient-wise) |
| 조기종료 | patience=5, monitor=val_auroc |
| 데이터셋 | NIH ChestX-ray14 (112,120장 → 111,979장 전처리 후) |

## 14개 질환(순서대로)

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

## 데이터셋

전처리 완료된 데이터셋: [MouGam/nih-processed-dataset](https://huggingface.co/datasets/MouGam/nih-processed-dataset)

다운로드:

```bash
pip install huggingface_hub[hf_transfer]
HF_HUB_ENABLE_HF_TRANSFER=1 python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='MouGam/nih-processed-dataset', repo_type='dataset', local_dir='./data_download')
"
mkdir -p data
tar xzf data_download/nih-processed-dataset.tar.gz -C data
```

### 전처리 파이프라인

1. 품질 필터링 (mean/std 기반, 소아 희귀질환 표본 보존)
2. CLAHE (clipLimit=2.0, tileGridSize=8x8)
3. Resize 224x224 (Bilinear)
4. Grayscale → 3채널 RGB
5. Train/Test Split (85:15, Patient-wise)
6. 5-Fold GroupKFold (Patient-wise)
7. Multi-hot Encoding (14개 질환 알파벳 순)

## 트레이닝

상세 가이드: [docs/training_guide.md](docs/training_guide.md)

### 빠른 시작

```bash
# gamma=0,1,2 전체 자동 실행
bash scripts/run_training.sh

# 개별 실행
python scripts/chestxray_train.py \
  --train_csv data/available/train.csv \
  --test_csv data/available/test.csv \
  --image_dir data/available/images \
  --output_dir outputs \
  --gammas 0 1 2
```

## 원본 데이터셋 출처

- **NIH ChestX-ray14**: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **License**: CC0 1.0 (Public Domain)

```bibtex
@inproceedings{wang2017chestx,
  title={ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks},
  author={Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald M},
  booktitle={CVPR},
  year={2017}
}
```
