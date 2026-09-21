"""
Python-клиент для Sentinel-Q HTTP-сервиса.

Использование:
    client = SentinelClient("http://localhost:8000")
    verdict = client.detect(image_bytes, predicted_label=3)
    if verdict["is_adversarial"]:
        print("Атака!")

    # Пакет
    results = client.detect_batch([b1, b2, b3], [1, 2, 3])
"""

from __future__ import annotations
import base64
from typing import Optional

try:
    import requests
except ImportError:
    raise ImportError("Установите: pip install requests")


class SentinelClient:

    def __init__(self, base_url: str = "http://localhost:8000",
                 timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def healthz(self) -> dict:
        r = requests.get(f"{self.base_url}/healthz", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def profile(self) -> dict:
        r = requests.get(f"{self.base_url}/v1/profile", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def detect(self, sample: bytes, predicted_label: int,
               combine: Optional[str] = None,
               threshold: Optional[float] = None) -> dict:
        payload = {
            "sample_b64": base64.b64encode(sample).decode("ascii"),
            "predicted_label": int(predicted_label),
        }
        if combine is not None:
            payload["combine"] = combine
        if threshold is not None:
            payload["threshold"] = threshold
        r = requests.post(f"{self.base_url}/v1/detect",
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def detect_batch(self, samples: list, labels: list,
                     combine: Optional[str] = None,
                     threshold: Optional[float] = None) -> dict:
        payload = {
            "samples_b64": [base64.b64encode(s).decode("ascii") for s in samples],
            "predicted_labels": [int(y) for y in labels],
        }
        if combine is not None:
            payload["combine"] = combine
        if threshold is not None:
            payload["threshold"] = threshold
        r = requests.post(f"{self.base_url}/v1/detect/batch",
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()