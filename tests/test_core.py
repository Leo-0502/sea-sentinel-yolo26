import csv
import tempfile
import unittest
from pathlib import Path

from sea_sentinel.core import BoatIDManager, DetectionEventLogger, resolve_target_classes


class TestBoatIDManager(unittest.TestCase):
    def test_ids_are_stable_and_reused(self):
        manager = BoatIDManager()
        self.assertEqual(manager.get_display_id(100), 1)
        self.assertEqual(manager.get_display_id(100), 1)
        self.assertEqual(manager.get_display_id(200), 2)
        manager.update_active_track_ids([200])
        self.assertEqual(manager.get_display_id(300), 1)

    def test_capacity_is_enforced(self):
        manager = BoatIDManager(max_display_id=2)
        manager.get_display_id(1)
        manager.get_display_id(2)
        with self.assertRaises(RuntimeError):
            manager.get_display_id(3)


class TestClassResolution(unittest.TestCase):
    def test_coco_boat_class_is_selected(self):
        self.assertEqual(resolve_target_classes({0: "person", 8: "boat", 9: "traffic light"}), [8])

    def test_custom_single_class_uses_all_classes(self):
        self.assertIsNone(resolve_target_classes({0: "vessel"}))

    def test_ship_synonyms_are_supported(self):
        self.assertEqual(resolve_target_classes(["person", "ship", "aircraft"]), [1])


class TestEventLogger(unittest.TestCase):
    def test_writes_header_and_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            logger = DetectionEventLogger(path)
            logger.write("entered", "camera:0", 2, 0.918)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], list(DetectionEventLogger.HEADER))
            self.assertEqual(rows[1][1:], ["entered", "camera:0", "2", "0.918"])


if __name__ == "__main__":
    unittest.main()
