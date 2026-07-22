FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FALL_YOLO_MODEL=/home/user/app/runs/detect/fall_detection_v3/weights/best.pt \
    FALL_ENABLE_VIDEOMAE=false \
    FALL_PROCESS_EVERY_N_FRAMES=3 \
    FALL_INPUT_SIZE=480
WORKDIR /home/user/app

COPY --chown=user requirements-space.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchvision==0.23.0 && \
    pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user . .
EXPOSE 7860
CMD ["python", "web_app.py"]
