import os
from pathlib import Path

from ultralytics import YOLO

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = _ROOT / "runs" / "detect" / "fall_detection_v3" / "weights" / "best.pt"
MODEL_PATH = Path(os.environ.get("FALL_YOLO_MODEL", str(_DEFAULT_MODEL))).expanduser()
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Fall YOLO model not found: {MODEL_PATH}. Set FALL_YOLO_MODEL to the weight path."
    )

model = YOLO(str(MODEL_PATH))
model.to("cpu")


def detect_fall_yolo(frame):
    results = model(frame, verbose=False, device="cpu")
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            if "fall" in label.lower() and conf > 0.7:
                return 1
    return 0
