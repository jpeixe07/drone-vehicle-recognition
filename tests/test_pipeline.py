"""Coordinate, class-filtering, and report regressions without model downloads."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image
import torch

from vehicle_recognition.pipeline import bbox_iou, map_box_to_original, recognize_image
from vehicle_recognition.reporting import RunReporter, write_json
from vehicle_recognition.similarity_efficientnet import DroneVehicleClassifier


class PipelineTests(unittest.TestCase):
    def test_ninety_degree_rotation_maps_boxes_back(self):
        image = np.zeros((60, 100, 3), dtype=np.uint8)
        rotated, matrix = DroneVehicleClassifier.rotate_image_bound(image, 90, True)
        self.assertEqual(rotated.shape[:2], (100, 60))
        corners = np.array([[[10.0, 15.0], [30.0, 15.0], [30.0, 35.0], [10.0, 35.0]]])
        mapped = cv2.transform(corners, matrix)[0]
        box = [mapped[:, 0].min(), mapped[:, 1].min(), mapped[:, 0].max(), mapped[:, 1].max()]
        np.testing.assert_allclose(map_box_to_original(box, matrix, 100, 60), [10, 15, 30, 35])

    def test_filter_uses_checkpoint_names_and_clips_crop_margins(self):
        def box(class_id, coordinates):
            return SimpleNamespace(cls=torch.tensor(class_id), conf=torch.tensor(0.9), xyxy=torch.tensor([coordinates]))
        class Detector:
            names = {0: "person", 7: "car"}
            def predict(self, **kwargs):
                self.options = kwargs
                return [SimpleNamespace(boxes=[box(0, [1.0, 1.0, 9.0, 9.0]), box(7, [-2.0, 3.0, 12.0, 40.0])])]
        detector = Detector()
        classifier = DroneVehicleClassifier(device="cpu", detector=detector)
        results = classifier.detect_vehicle_crops(Image.new("RGB", (20, 30)))
        self.assertEqual(len(results), 1)
        self.assertEqual(detector.options["classes"], [7])
        self.assertEqual(results[0]["bbox"], [0, 3, 12, 30])
        self.assertEqual(results[0]["_crop"].size, (20, 30))

    def test_rotated_views_form_one_object_not_multiple_vehicle_predictions(self):
        class Classifier:
            rotate_image_bound = staticmethod(DroneVehicleClassifier.rotate_image_bound)
            def detect_vehicle_crops(self, image):
                width, height = image.size
                return [{"bbox": [0, 0, width, height], "_crop": image,
                         "detector_class": "car", "detection_confidence": 0.9}]
            def classify_crops(self, crops):
                return {"label": "a", "accepted": True, "observation_count": len(crops), "similarity": 0.9}
        results = recognize_image(Classifier(), Image.new("RGB", (100, 60)), [0, 90, 360])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["observation_count"], 2)
        self.assertEqual(bbox_iou(results[0]["bbox"], [0, 0, 100, 60]), 1)

    def test_empty_detections_remain_in_json_and_csv_has_a_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            reporter = RunReporter(directory, {"mode": "test"}, save_images=False)
            reporter.record({"image": "empty.jpg", "detections": []})
            summary = reporter.finish(0.1)
            self.assertEqual(summary["images"], 1)
            self.assertEqual(summary["detections"], 0)
            self.assertEqual(len(json.loads((directory / "predictions.json").read_text())["images"]), 1)
            self.assertTrue((directory / "predictions.csv").read_text().startswith("image,"))

    def test_failed_json_save_preserves_previous_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json(path, {"revision": "previous"})
            with self.assertRaises(ValueError):
                write_json(path, {"score": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"revision": "previous"})


if __name__ == "__main__":
    unittest.main()
