from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Iterable, Mapping


class BoatIDManager:
    """Map tracker IDs to compact display IDs and recycle inactive IDs."""

    def __init__(self, max_display_id: int = 99) -> None:
        if max_display_id < 1:
            raise ValueError("max_display_id must be positive")
        self.max_display_id = max_display_id
        self.track_id_to_display_id: dict[int, int] = {}
        self.available_display_ids: list[int] = []
        self.next_new_display_id = 1

    def get_display_id(self, track_id: int) -> int:
        if track_id in self.track_id_to_display_id:
            return self.track_id_to_display_id[track_id]
        used = set(self.track_id_to_display_id.values())
        if self.available_display_ids:
            display_id = self.available_display_ids.pop(0)
        else:
            display_id = self._next_available_id(used)
        self.track_id_to_display_id[track_id] = display_id
        return display_id

    def _next_available_id(self, used: set[int]) -> int:
        for _ in range(self.max_display_id):
            candidate = self.next_new_display_id
            self.next_new_display_id = candidate % self.max_display_id + 1
            if candidate not in used:
                return candidate
        raise RuntimeError("No display IDs are available")

    def update_active_track_ids(self, active_track_ids: Iterable[int]) -> None:
        active = set(active_track_ids)
        for track_id in set(self.track_id_to_display_id) - active:
            self.available_display_ids.append(self.track_id_to_display_id.pop(track_id))
        self.available_display_ids.sort()

    def clear(self) -> None:
        self.track_id_to_display_id.clear()
        self.available_display_ids.clear()
        self.next_new_display_id = 1


def resolve_target_classes(names: Mapping[int, str] | list[str]) -> list[int] | None:
    """Return vessel-like class IDs, or all classes for a custom model."""
    items = names.items() if isinstance(names, Mapping) else enumerate(names)
    normalized = {int(index): str(name).strip().lower() for index, name in items}
    if len(normalized) <= 1:
        return None
    keywords = ("boat", "ship", "vessel", "watercraft", "船", "舰", "艇")
    matches = [index for index, name in normalized.items() if any(word in name for word in keywords)]
    return matches or None


class DetectionEventLogger:
    """Append detection state transitions to a CSV audit log."""

    HEADER = ("timestamp", "event", "source", "boat_count", "max_confidence")

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, event: str, source: str, boat_count: int, max_confidence: float) -> None:
        with self._lock:
            is_new = not self.path.exists()
            with self.path.open("a", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                if is_new:
                    writer.writerow(self.HEADER)
                writer.writerow([datetime.now().isoformat(timespec="seconds"), event, source, boat_count, f"{max_confidence:.3f}"])
