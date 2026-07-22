"""Hugging Face Spaces browser-webcam entry point."""

import os

# ZeroGPU is selected as the free Space host, but realtime inference stays on CPU.
# Do not add @spaces.GPU to the per-frame WebRTC handler.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FALL_ENABLE_VIDEOMAE", "false")
os.environ.setdefault("FALL_INPUT_SIZE", "384")

from fastrtc import Stream, VideoStreamHandler, get_turn_credentials
import spaces

from detector.webrtc_detector import WebRTCFallDetector


RESPONSIVE_CSS = """
html, body, gradio-app {
    height: auto !important;
    min-height: 100% !important;
    overflow-y: auto !important;
    scrollbar-gutter: stable;
}

.gradio-container, main.fillable {
    height: auto !important;
    min-height: 100vh !important;
    overflow: visible !important;
}

.video-container {
    width: min(100%, 900px, calc(65vh * 4 / 3)) !important;
    max-width: 900px !important;
    height: auto !important;
    max-height: 65vh !important;
    aspect-ratio: 4 / 3 !important;
    overflow: visible !important;
    margin-inline: auto !important;
}

.video-container,
.video-container * {
    transition: none !important;
    animation: none !important;
}

.video-container .wrap {
    position: relative !important;
    inset: auto !important;
    display: flex !important;
    flex-direction: column !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 0 !important;
}

.video-container video {
    position: static !important;
    width: 100% !important;
    height: auto !important;
    max-height: min(65vh, 560px) !important;
    aspect-ratio: 4 / 3 !important;
    object-fit: contain !important;
    background: #000;
}

.video-container .button-wrap {
    position: static !important;
    inset: auto !important;
    transform: none !important;
    flex: 0 0 auto !important;
    justify-content: center !important;
    margin: 0.75rem auto 1rem !important;
}

footer {
    position: static !important;
    flex: 0 0 auto !important;
    margin-top: 1rem !important;
}

@media (max-width: 640px) {
    .video-container .button-wrap { width: 100% !important; }
}
"""


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
        "full_screen": False,
    },
)
demo = stream.ui
demo.css = RESPONSIVE_CSS

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        ssr_mode=False,
    )
