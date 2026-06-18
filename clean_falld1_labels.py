"""
falld_1 라벨 정제: 클래스 0(fall)만 유지하고 1, 2 제거
- 원본은 _original_backup 폴더에 자동 백업
- 정제 후 모든 374장이 학습에 사용 가능
- not fallen 이미지의 라벨은 비워짐 (negative sample로 작용)
"""
import shutil
from pathlib import Path

DATASET_ROOT = Path(r"data\falld_1\fall_dataset")
TRAIN_LBL = DATASET_ROOT / "labels" / "train"
VAL_LBL = DATASET_ROOT / "labels" / "val"


def clean_labels(label_dir, name):
    backup_dir = label_dir.parent / f"{label_dir.name}_original_backup"
    if not backup_dir.exists():
        print(f"  원본 백업 생성: {backup_dir}")
        shutil.copytree(label_dir, backup_dir)
    else:
        print(f"  백업 이미 존재 (스킵): {backup_dir}")

    files = list(label_dir.glob("*.txt"))
    print(f"\n[{name}] {len(files)}개 라벨 파일 정제 중...")

    line_kept = 0
    line_removed = 0
    files_with_fall = 0
    files_negative = 0   # 빈 라벨 (negative sample)

    for txt in files:
        with open(txt, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for ln in lines:
            parts = ln.strip().split()
            if not parts:
                continue
            try:
                cls = int(parts[0])
                if cls == 0:
                    new_lines.append(ln if ln.endswith("\n") else ln + "\n")
                    line_kept += 1
                else:
                    line_removed += 1
            except ValueError:
                continue

        with open(txt, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        if new_lines:
            files_with_fall += 1
        else:
            files_negative += 1

    print(f"  유지된 fall 박스:        {line_kept}개")
    print(f"  제거된 박스(클래스1,2): {line_removed}개")
    print(f"  fall 라벨 보유 파일:    {files_with_fall}개")
    print(f"  빈 라벨(negative):     {files_negative}개")


def main():
    print("=" * 70)
    print("falld_1 라벨 정제 - 클래스 0(fall)만 유지")
    print("=" * 70)

    clean_labels(TRAIN_LBL, "train")
    clean_labels(VAL_LBL, "val")

    print("\n정제 완료. 이제 'python train_yolo.py' 다시 실행하세요.")
    print("이번엔 corrupt 경고 없이 모든 이미지가 학습에 사용됩니다.")
    print("\n원본 라벨로 되돌리려면 _original_backup 폴더 내용으로 복원.")


if __name__ == "__main__":
    main()
