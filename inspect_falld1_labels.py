"""
falld_1 데이터셋의 라벨 클래스 분포 진단
- 각 클래스 ID별 라벨 개수
- 클래스별 파일 샘플
- fall vs not_fall 파일명 패턴과 클래스 ID의 관계
"""
import os
from pathlib import Path
from collections import defaultdict

DATASET_ROOT = Path(r"data\falld_1\fall_dataset")
TRAIN_LBL = DATASET_ROOT / "labels" / "train"
VAL_LBL = DATASET_ROOT / "labels" / "val"


def scan(label_dir, name):
    if not label_dir.exists():
        print(f"  [없음] {label_dir}")
        return {}, defaultdict(list)

    class_count = defaultdict(int)
    class_files = defaultdict(list)  # class_id -> [filename, ...]
    fall_classes = defaultdict(int)  # fall*.txt에 나타나는 클래스
    notfall_classes = defaultdict(int)

    txt_files = list(label_dir.glob("*.txt"))
    print(f"\n[{name}] 라벨 파일 {len(txt_files)}개")

    for txt in txt_files:
        try:
            with open(txt, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            continue
        is_fall = "fall" in txt.name.lower() and "not" not in txt.name.lower()
        for ln in lines:
            parts = ln.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
            except ValueError:
                continue
            class_count[cls] += 1
            if len(class_files[cls]) < 5:
                class_files[cls].append(txt.name)
            if is_fall:
                fall_classes[cls] += 1
            else:
                notfall_classes[cls] += 1

    print(f"  클래스별 라벨 수:")
    for cls in sorted(class_count.keys()):
        print(f"    클래스 {cls}: {class_count[cls]}개  (샘플 파일: {class_files[cls][:3]})")
    print(f"  fall*.txt 안의 클래스 분포: {dict(fall_classes)}")
    print(f"  not fallen*.txt 안의 클래스 분포: {dict(notfall_classes)}")
    return class_count, class_files


def main():
    print("=" * 70)
    print("falld_1 라벨 클래스 진단")
    print("=" * 70)

    scan(TRAIN_LBL, "train")
    scan(VAL_LBL, "val")

    print("\n=" * 70)
    print("해석 가이드:")
    print("  - 클래스 0: 보통 첫 번째 (fall?)")
    print("  - fall*.txt에 다른 클래스가 있다면 → 그게 진짜 'fall' 클래스일 수도")
    print("  - not fallen*.txt가 클래스 1 또는 2를 가지면 → 그게 'not_fall' 클래스")
    print("=" * 70)


if __name__ == "__main__":
    main()
