"""Optional VideoMAE detector, lazily loaded for CPU deployments."""

import cv2
import torch
from transformers import AutoImageProcessor, AutoModelForVideoClassification

MODEL_ID = "yadvender12/videomae-base-finetuned-kinetics-finetuned-fall-detect"
processor = None
model = None
frame_buffer = []
BUFFER_SIZE = 16
frame_count = 0
last_score = 0


def load_videomae():
    global processor, model
    if model is None:
        print("Loading optional VideoMAE model...")
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
        model.eval()
        print("VideoMAE loaded")
    return processor, model


def detect_fall_videomae(frame):
    global frame_count, last_score
    frame_count += 1
    resized = cv2.resize(frame, (224, 224))
    frame_buffer.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    if len(frame_buffer) > BUFFER_SIZE:
        frame_buffer.pop(0)
    if frame_count % 10 != 0:
        return last_score
    if len(frame_buffer) < BUFFER_SIZE:
        return 0

    current_processor, current_model = load_videomae()
    inputs = current_processor(list(frame_buffer), return_tensors="pt")
    with torch.no_grad():
        logits = current_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        predicted = torch.argmax(logits, dim=-1).item()
        confidence = probs[0][predicted].item()
        label = current_model.config.id2label[predicted]
    if label == "FallDown" and confidence >= 0.6:
        last_score = 2
    elif label == "LyingDown" and confidence >= 0.85:
        last_score = 1
    else:
        last_score = 0
    return last_score
