---
title: Real-time Fall Detection WebRTC
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.49.1
app_file: web_app.py
python_version: '3.10.13'
suggested_hardware: zero-a10g
---

# Browser WebRTC fall detection

`main.py` remains the local-camera application. `web_app.py` receives the visitor's
browser webcam through FastRTC/WebRTC on Hugging Face Spaces.

## Deploy to Hugging Face Spaces

1. Create a **Gradio Space**, select **ZeroGPU**, and push this repository.
2. In **Settings > Variables and secrets**, configure `.env.example`. Set
   `FALL_ALERT_SPRING_URL` to the public HTTPS base URL of woori_link Spring Boot.
   Leave it empty to disable outbound alerts.
   Add `HF_TOKEN` as a Space **Secret** so FastRTC can obtain hosted TURN credentials.
3. Keep `FALL_ENABLE_VIDEOMAE=false` on ZeroGPU. Enabling it downloads a
   large transformer at runtime and increases cold-start time and latency.

The real-time detector deliberately runs on CPU and does not use `@spaces.GPU`, so
every WebRTC frame does not request a ZeroGPU allocation. `zero-a10g` in the metadata
is the matching ZeroGPU suggestion; hardware selection itself is done in Space settings.

Alerts use `POST {FALL_ALERT_SPRING_URL}/api/alerts/fall`. An optional
`FALL_ALERT_API_KEY` is sent as a Bearer token. Alerts require consecutive positive
inference frames and are rate-limited by `FALL_ALERT_COOLDOWN_SEC`.

## Run locally

```bash
python -m pip install -r requirements-space.txt
python web_app.py
```

Open `http://localhost:7860`, allow camera access, and press Start. Docker:

```bash
docker build -t fall-webrtc .
docker run --rm -p 7860:7860 --env-file .env fall-webrtc
```

## CPU tuning

- Increase `FALL_PROCESS_EVERY_N_FRAMES` from 3 to 4 or 5 if inference lags.
- `FALL_INPUT_SIZE` defaults to 384 for CPU inference; WebRTC output remains 640x480.
- `FALL_CONFIRM_FRAMES` controls consecutive sampled positives before alerting.
- `FALL_XGBOOST_THR` controls the XGBoost probability threshold.

The WebRTC app does not save webcam frames and permits one active inference worker
per Space instance to avoid memory and CPU contention. `Dockerfile` is retained only
for local Docker execution; Hugging Face builds from `requirements.txt` and `packages.txt`.
