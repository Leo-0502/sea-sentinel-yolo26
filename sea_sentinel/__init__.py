"""Core utilities for the Sea Sentinel desktop application."""

from .core import BoatIDManager, DetectionEventLogger, resolve_target_classes
from .model import ensure_model

__all__ = ["BoatIDManager", "DetectionEventLogger", "resolve_target_classes", "ensure_model"]
