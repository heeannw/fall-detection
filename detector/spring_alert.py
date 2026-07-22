"""Non-blocking Spring fall alert client for the WebRTC app."""

import logging
import os
import threading
import time

import requests

log = logging.getLogger(__name__)


class SpringFallNotifier:
    def __init__(self):
        self.base_url = os.getenv("FALL_ALERT_SPRING_URL", "").rstrip("/")
        self.senior_id = int(os.getenv("FALL_SENIOR_ID", "53"))
        self.api_key = os.getenv("FALL_ALERT_API_KEY", "")
        self.cooldown = float(os.getenv("FALL_ALERT_COOLDOWN_SEC", "30"))
        self.timeout = float(os.getenv("FALL_ALERT_TIMEOUT_SEC", "5"))
        self._last_sent = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self):
        return bool(self.base_url)

    def notify_async(self, details):
        if not self.enabled:
            return False
        with self._lock:
            now = time.monotonic()
            if now - self._last_sent < self.cooldown:
                return False
            self._last_sent = now
        threading.Thread(target=self._send, args=(details,), daemon=True).start()
        return True

    def _send(self, details):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "seniorId": self.senior_id,
            "score": int(details.get("score", 0)),
            "imageUrl": None,
            "imageAccessUrl": None,
            "notifyGuardian": True,
            "notifyWelfare": False,
            "escalationRequired": bool(details.get("persistent", False)),
            "fallDetails": details,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/alerts/fall",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            log.info("Spring fall alert sent for seniorId=%s", self.senior_id)
        except requests.RequestException:
            log.exception("Spring fall alert failed")
