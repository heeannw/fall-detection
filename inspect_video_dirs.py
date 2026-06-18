"""
옵션 B 영상 데이터셋 폴더 구조 파악용 스크립트
- 각 데이터셋 폴더의 트리 구조 + 영상 파일 분포 확인
- 낙상/정상 구분 방식(폴더명/파일명 패턴) 파악
"""
import os

DIRS = [
    (r"data\falld_1", "Le2i (falldataset-imvia)"),
    (r"data\falld_2", "Fall Video Dataset (payutch)"),
    (r"data\falld_3", "Fall Detection Dataset (uttejkumarkandagatla)"),
]

VIDEO_EXT = {".avi", ".mp4", ".mov", ".mkv", ".wmv"}
ANNO_EXT = {".txt", ".csv", ".json", ".xml"}


def scan_dir(root):
    """폴더 내 모든 파일을 확장자별로 집계 + 트리 구조 + 키워드 분포"""
    if not os.path.exists(root):
        print(f"  ❌ 존재하지 않음: {root}")
        return

    # 트리 구조 (3레벨까지)
    print("\n[폴더 트리 (3레벨)]")
    for cur, dirs, files in os.walk(root):
        level = cur.replace(root, "").count(os.sep)
        if level > 2:
            continue
        indent = "  " * level
        print(f"{indent}{os.path.basename(cur) or os.path.basename(root)}/")
        if level <= 2:
            shown = files[:5]
            for f in shown:
                print(f"{indent}  {f}")
            if len(files) > 5:
                print(f"{indent}  ...({len(files)}개 더)")

    # 확장자별 집계
    ext_count = {}
    video_files = []
    anno_files = []
    for cur, dirs, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            ext_count[ext] = ext_count.get(ext, 0) + 1
            if ext in VIDEO_EXT:
                video_files.append(os.path.join(cur, f))
            if ext in ANNO_EXT:
                anno_files.append(os.path.join(cur, f))

    print(f"\n[확장자별 파일 수]")
    for ext, cnt in sorted(ext_count.items(), key=lambda x: -x[1]):
        print(f"  {ext or '(no-ext)'}: {cnt}")

    print(f"\n[영상 파일 총 {len(video_files)}개]")
    for v in video_files[:10]:
        rel = os.path.relpath(v, root)
        size_mb = os.path.getsize(v) / 1024 / 1024
        print(f"  {rel}  ({size_mb:.1f} MB)")
    if len(video_files) > 10:
        print(f"  ... 외 {len(video_files) - 10}개")

    # 키워드 분포 (낙상/정상 구분 후보 키워드)
    print(f"\n[키워드 분포 - 영상 파일 경로 기준]")
    keywords = ["fall", "Fall", "FALL", "adl", "ADL", "Adl",
                "normal", "Normal", "not_fall", "notfall", "NoFall",
                "daily", "Daily", "non-fall", "nonfall"]
    for kw in keywords:
        matched = [v for v in video_files if kw in v]
        if matched:
            print(f"  '{kw}' 포함: {len(matched)}개 (예: {os.path.relpath(matched[0], root)})")

    # annotation 파일
    if anno_files:
        print(f"\n[Annotation 파일 (최대 5개)]")
        for a in anno_files[:5]:
            rel = os.path.relpath(a, root)
            size_kb = os.path.getsize(a) / 1024
            print(f"  {rel}  ({size_kb:.1f} KB)")
            # 첫 5줄 미리보기
            try:
                with open(a, "r", encoding="utf-8", errors="ignore") as fh:
                    lines = fh.readlines()[:5]
                for ln in lines:
                    print(f"      | {ln.rstrip()}")
            except Exception as e:
                print(f"      (읽기 실패: {e})")


for path, name in DIRS:
    print("\n" + "=" * 70)
    print(f"== {name}")
    print(f"== {path}")
    print("=" * 70)
    scan_dir(path)
