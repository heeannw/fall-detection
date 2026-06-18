"""
Phase 1 - 길 2: YOLO 재학습
- falld_1 (fall_dataset) 데이터셋으로 YOLOv8 재학습
- data.yaml 자동 생성 (없을 경우)
- train/val 분할 자동 처리 (val 폴더 없으면 train의 20%를 val로)
- 학습 완료 후 새 best.pt 위치 출력
"""
import os
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

DATASET_ROOT = Path(r"data\falld_1\fall_dataset")
YAML_PATH = DATASET_ROOT / "data.yaml"

IMAGES_DIR = DATASET_ROOT / "images"
LABELS_DIR = DATASET_ROOT / "labels"
TRAIN_IMG_DIR = IMAGES_DIR / "train"
VAL_IMG_DIR = IMAGES_DIR / "val"
TRAIN_LBL_DIR = LABELS_DIR / "train"
VAL_LBL_DIR = LABELS_DIR / "val"


def ensure_val_split(val_ratio=0.2, seed=42):
    """val 폴더가 없으면 train에서 일부를 val로 분리 (한 번만 실행)"""
    if VAL_IMG_DIR.exists() and any(VAL_IMG_DIR.iterdir()):
        print(f"  val 폴더 이미 존재: {VAL_IMG_DIR}")
        n_train = sum(1 for _ in TRAIN_IMG_DIR.glob("*"))
        n_val = sum(1 for _ in VAL_IMG_DIR.glob("*"))
        print(f"  train: {n_train}장, val: {n_val}장")
        return

    if not TRAIN_IMG_DIR.exists():
        raise FileNotFoundError(f"train 폴더가 없음: {TRAIN_IMG_DIR}")

    VAL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    VAL_LBL_DIR.mkdir(parents=True, exist_ok=True)

    # train의 이미지 파일 목록 수집
    img_files = []
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        img_files.extend(TRAIN_IMG_DIR.glob(f"*{ext}"))
        img_files.extend(TRAIN_IMG_DIR.glob(f"*{ext.upper()}"))

    if not img_files:
        raise FileNotFoundError(f"train 폴더에 이미지 없음: {TRAIN_IMG_DIR}")

    random.Random(seed).shuffle(img_files)
    n_val = max(1, int(len(img_files) * val_ratio))
    val_files = img_files[:n_val]

    moved = 0
    for img_path in val_files:
        # 이미지 이동
        dst_img = VAL_IMG_DIR / img_path.name
        shutil.move(str(img_path), str(dst_img))
        # 대응 라벨 이동
        lbl_name = img_path.stem + ".txt"
        src_lbl = TRAIN_LBL_DIR / lbl_name
        if src_lbl.exists():
            dst_lbl = VAL_LBL_DIR / lbl_name
            shutil.move(str(src_lbl), str(dst_lbl))
        moved += 1

    print(f"  val 분할 완료: train {len(img_files)-moved}장 / val {moved}장")


def ensure_yaml():
    """data.yaml이 없으면 자동 생성"""
    if YAML_PATH.exists():
        print(f"  data.yaml 이미 존재: {YAML_PATH}")
        return YAML_PATH

    config = {
        "path": str(DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["fall"],
    }
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True)
    print(f"  data.yaml 생성: {YAML_PATH}")
    return YAML_PATH


def main():
    print("=" * 70)
    print("Phase 1 - 길 2: YOLO 재학습 (falld_1)")
    print("=" * 70)
    print(f"  데이터셋 루트: {DATASET_ROOT}")
    if not DATASET_ROOT.exists():
        print(f"\n  [에러] 데이터셋 폴더 없음: {DATASET_ROOT}")
        return

    # 1) train/val 분할 확인 + 자동 분리
    ensure_val_split(val_ratio=0.2, seed=42)

    # 2) data.yaml 확보
    yaml_path = ensure_yaml()

    # 3) 학습 시작
    print(f"\n  학습 시작 - epochs=50, imgsz=640, patience=30")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=50,            # 15 → 50 (충분한 학습)
        imgsz=640,            # 416 → 640 (정확도 향상, 약간 느려짐)
        batch=16,
        name="fall_detection_v3",  # 이전 망가진 v2와 분리
        patience=30,          # 너무 짧으면 1 epoch 후 끝나버림 (10 → 30)
        workers=4,
        verbose=True,
    )

    print("\n학습 완료!")
    print(f"새 모델 위치: runs/detect/fall_detection_v3/weights/best.pt")
    print(f"\n다음 단계: python apply_new_yolo.py  (yolo_detector.py 경로 자동 업데이트)")


if __name__ == "__main__":
    main()
