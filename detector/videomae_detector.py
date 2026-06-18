import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForVideoClassification
import cv2

print("VideoMAE 모델 로딩 중...")
processor = AutoImageProcessor.from_pretrained('yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect')
model = AutoModelForVideoClassification.from_pretrained('yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect')
model.eval()
print("VideoMAE 로딩 완료!")

frame_buffer = []
BUFFER_SIZE = 16
frame_count = 0
last_score = 0

def detect_fall_videomae(frame):
    global frame_buffer, frame_count, last_score

    frame_count += 1

    resized = cv2.resize(frame, (224, 224))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    frame_buffer.append(rgb)

    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)

    if frame_count % 10 != 0:
        return last_score

    if len(frame_buffer) < BUFFER_SIZE:
        return 0

    inputs = processor(list(frame_buffer), return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        predicted = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted].item()
        label = model.config.id2label[predicted]

    print(f"VideoMAE → 라벨: {label}, 신뢰도: {confidence:.2f}")

    # Fix #6+#9: FallDown(+2), LyingDown 신뢰도 0.85+ 매우 강할 때만 +1
    # 0.7→0.85: 정상 영상의 일상 lying(폰 보기/자기)에서 자주 발동되던 문제 해결
    if label == "FallDown" and confidence >= 0.6:
        last_score = 2
    elif label == "LyingDown" and confidence >= 0.85:
        last_score = 1
    else:
        last_score = 0

    return last_score