"""
PC에 연결된 카메라 인덱스 검색
실행: python list_cameras.py
"""
import cv2

print("=" * 50)
print("연결된 카메라 검색 중...")
print("=" * 50)

found = []
for idx in range(5):   # 0~4 인덱스 확인
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  [{idx}] ✓ 작동함 - 해상도 {w}x{h}")
            found.append(idx)
        else:
            print(f"  [{idx}] ✗ 열렸지만 프레임 못 읽음")
        cap.release()
    else:
        print(f"  [{idx}] - 없음")

print("=" * 50)
if found:
    print(f"\n작동하는 카메라 인덱스: {found}")
    print(f"\n외장 카메라(Logitech 등) 사용 시:")
    print(f"  $env:FALL_CAMERA_INDEX = \"{found[-1]}\"  # 보통 마지막 인덱스가 외장")
    print(f"  python -m uvicorn main:app --host 0.0.0.0 --port 8000")
else:
    print("\n[에러] 작동하는 카메라가 없음. USB 연결 확인.")
