"""
Sentinel-Q — model-free adversarial detection.

Public API:
    QorbDescriptor      — ядро: метрика w, CV, H
    ReferenceProfile    — эталонные распределения
    SentinelDetector    — детекция
    AttackClassifier    — классификация типа атаки
    StreamMonitor       — мониторинг потока
"""

from .core import QorbDescriptor
from .profile import ReferenceProfile
from .detector import SentinelDetector
from .classifier import AttackClassifier
from .streaming import StreamMonitor

__version__ = "0.1.0"
__all__ = [
    "QorbDescriptor",
    "ReferenceProfile",
    "SentinelDetector",
    "AttackClassifier",
    "StreamMonitor",
]