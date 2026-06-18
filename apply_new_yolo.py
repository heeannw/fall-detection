"""
train_yolo.py 학습 후 새 best.pt를 yolo_detector.py에 자동 반영
- 기존 yolo_detector.py 백업 (.bak)
- 새 best.pt 경로 검증 후 교체
- 동시에 신뢰도 임계값(현재 0.85)도 학습 결과에 맞춰 조정 가능
"""
import os
import re
import shutil
from pathlib import Path

YOLO_DETECTOR_PATH = r"C:\fall-detection\detector\yolo_detector.py"
NEW_BEST_PATH = r"C:\fall-detection\runs\detect\fall_detection_v3\weights\best.pt"


def main():
    if not os.path.exists(NEW_BEST_PATH):
        print(f"[에러] 새 모델 없음: {NEW_BEST_PATH}")
        print("  먼저 'python train_yolo.py' 실행하세요.")
        return

    # 백업
    backup = YOLO_DETECTOR_PATH + ".bak"
    shutil.copy2(YOLO_DETECTOR_PATH, backup)
    print(f"백업: {backup}")

    # yolo_detector.py 읽기
    with open(YOLO_DETECTOR_PATH, "r", encoding="utf-8") as f:
        src = f.read()

    # YOLO 모델 경로 교체 (정규식 replacement는 백슬래시 escape 문제 있어서 string.replace 사용)
    old_path_pattern = r'YOLO\(r?["\'][^"\']*best\.pt["\']\)'
    new_yolo_line = f'YOLO(r"{NEW_BEST_PATH}")'
    match = re.search(old_path_pattern, src)
    if not match:
        print("[경고] YOLO(...) 라인을 찾지 못함")
        return
    new_src = src.replace(match.group(0), new_yolo_line)
    print(f"YOLO 모델 경로 교체: {NEW_BEST_PATH}")

    # 신뢰도 임계값 0.85 → 0.5 로 낮춤 (재학습된 YOLO는 신뢰할 만하므로)
    # 단, 영상 평가에서 효과 검증 후 다시 조정 가능
    new_src = re.sub(
        r'conf > 0\.\d+', 'conf > 0.5', new_src
    )
    print("YOLO 신뢰도 임계값: 0.85 → 0.5 (재학습 후 더 엄격하지 않게)")

    with open(YOLO_DETECTOR_PATH, "w", encoding="utf-8") as f:
        f.write(new_src)
    print(f"업데이트 완료: {YOLO_DETECTOR_PATH}")
    print("\n다음 단계:")
    print("  1) Remove-Item -Recurse -Force C:\\fall-detection\\detector\\__pycache__")
    print("  2) python evaluate_video.py")


if __name__ == "__main__":
    main()
