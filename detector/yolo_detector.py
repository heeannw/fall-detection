from ultralytics import YOLO
import numpy as np

model = YOLO(r"C:\fall-detection\runs\detect\fall_detection_v3\weights\best.pt")

def detect_fall_yolo(frame):
    results = model(frame, verbose=False)
    
    for result in results:
        if result.boxes is None:
            continue
        
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            
            # 임계값 조정 (0.5→0.7): 새 YOLO가 정상 영상에 fall 오출력 줄이기 위해
            if "fall" in label.lower() and conf > 0.7:
                return 1
    
    return 0