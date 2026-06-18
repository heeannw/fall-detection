"""
tune_thresholds.py 실행 결과(tune_results.json)에서 최적 조합을 읽어
mediapipe_detector.py의 디폴트 임계값을 자동으로 업데이트한다.
- 기존 파일은 .bak2로 백업 후 교체
- main.py / evaluate_video.py는 공용 변수를 import해서 쓰므로 자동 반영됨
"""
import json
import os
import re
import shutil
import sys

DETECTOR_PATH = r"C:\fall-detection\detector\mediapipe_detector.py"
RESULTS_PATH = r"C:\fall-detection\tune_results.json"


def main():
    if not os.path.exists(RESULTS_PATH):
        print(f"[에러] {RESULTS_PATH} 없음. 먼저 'python tune_thresholds.py' 실행하세요.")
        sys.exit(1)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    best_combo = data["best"]["combo"]
    best_metrics = data["best"]["metrics"]
    baseline_metrics = data["baseline"]["metrics"]

    print("=" * 60)
    print("최적 조합 적용")
    print("=" * 60)
    print(f"  베이스라인: F1={baseline_metrics['f1']} acc={baseline_metrics['acc']} "
          f"sens={baseline_metrics['sens']} spec={baseline_metrics['spec']} prec={baseline_metrics['prec']}")
    print(f"  최적조합  : F1={best_metrics['f1']} acc={best_metrics['acc']} "
          f"sens={best_metrics['sens']} spec={best_metrics['spec']} prec={best_metrics['prec']}")
    print(f"\n  적용할 임계값:")
    for k, v in best_combo.items():
        print(f"    {k} = {v}")

    # 백업
    backup_path = DETECTOR_PATH + ".bak2"
    shutil.copy2(DETECTOR_PATH, backup_path)
    print(f"\n  백업: {backup_path}")

    # 파일 읽기
    with open(DETECTOR_PATH, "r", encoding="utf-8") as f:
        src = f.read()

    # 각 변수 라인을 정규식으로 찾아 새 값으로 교체
    updated = src
    for name, value in best_combo.items():
        # 예: LYING_HORIZONTAL_RATIO = 0.55 → LYING_HORIZONTAL_RATIO = 0.50
        if isinstance(value, float):
            new_line_value = f"{value}"
        else:
            new_line_value = f"{value}"
        pattern = rf"^{re.escape(name)}\s*=\s*[^\n#]+"
        replacement = f"{name} = {new_line_value}"
        new_text, count = re.subn(pattern, replacement, updated, flags=re.MULTILINE)
        if count == 0:
            print(f"  [경고] {name} 라인을 찾지 못함 (건너뜀)")
        else:
            updated = new_text
            print(f"  {name} → {value} 적용")

    # 저장
    with open(DETECTOR_PATH, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"\n  업데이트 완료: {DETECTOR_PATH}")
    print(f"\n  다음 단계: python evaluate_video.py  (전체 샘플로 재측정)")


if __name__ == "__main__":
    main()
