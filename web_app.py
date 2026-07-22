"""Hugging Face Spaces browser-webcam entry point."""

import os

# ZeroGPU is selected as the free Space host, but realtime inference stays on CPU.
# Do not add @spaces.GPU to the per-frame WebRTC handler.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FALL_ENABLE_VIDEOMAE", "false")

from fastrtc import Stream, VideoStreamHandler, get_turn_credentials
import spaces

from detector.webrtc_detector import WebRTCFallDetector


@spaces.GPU
def zerogpu_probe():
    """ZeroGPU startup probe; never used by the realtime WebRTC handler."""
    return "ZeroGPU ready"


detector = WebRTCFallDetector()
stream = Stream(
    handler=VideoStreamHandler(detector, skip_frames=True),
    modality="video",
    mode="send-receive",
    concurrency_limit=1,
    time_limit=int(os.getenv("FALL_SESSION_TIME_LIMIT_SEC", "900")),
    rtc_configuration=get_turn_credentials,
    ui_args={
        "title": "Real-time Fall Detection",
        "subtitle": "Browser webcam / MediaPipe + YOLO + XGBoost",
    },
)
demo = stream.ui

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
