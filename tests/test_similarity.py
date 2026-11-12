"""Regression tests for recognition semantics and gallery compatibility."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from vehicle_recognition.similarity_efficientnet import (
    DroneVehicleClassifier, VehicleReferenceDB, as_rgb_image, gallery_manifest,
)


class SimilarityTests(unittest.TestCase):
    def test_class_score_keeps_distinct_views_instead_of_averaging_references(self):
        database = VehicleReferenceDB(threshold=0.8)
        database.add("model_a", np.array([1, 0]), "front.jpg")
        database.add("model_a", np.array([-1, 0]), "back.jpg")
        database.add("model_b", np.array([0.6, 0.8]), "other.jpg")
        result = database.aggregate(np.array([[3, 0]]))
        self.assertEqual(result["label"], "model_a")
        self.assertAlmostEqual(result["similarity"], 1)
        self.assertEqual(result["top_candidates"][0]["reference"], "front.jpg")

    def test_rejection_retains_best_candidate_and_score(self):
        database = VehicleReferenceDB(threshold=0.8)
        database.add("known", np.array([1, 0]))
        result = database.aggregate(np.array([[0, 1]]))
        self.assertEqual(result["label"], "unknown")
        self.assertEqual(result["candidate_label"], "known")
        self.assertFalse(result["accepted"])
        self.assertEqual(result["similarity"], 0)

    def test_group_uses_all_class_scores_before_top_k_selection(self):
        database = VehicleReferenceDB(threshold=0.5)
        database.add("a", np.array([1, 0]))
        database.add("b", np.array([0, 1]))
        database.add("consistent", np.array([1, 1]))
        result = database.aggregate(np.eye(2), top_k=1)
        self.assertEqual(result["label"], "consistent")
        self.assertEqual(result["observation_count"], 2)
        self.assertAlmostEqual(result["similarity"], 2**-0.5, places=6)

    def test_missing_gallery_and_bad_vectors_fail_instead_of_becoming_unknown(self):
        with self.assertRaises(ValueError):
            VehicleReferenceDB().query(np.array([1, 0]))
        for value in [np.zeros(2), np.array([np.nan, 1]), np.ones((1, 2))]:
            with self.assertRaises(ValueError):
                VehicleReferenceDB().add("a", value)

    def test_bgr_frames_and_pil_images_have_consistent_colors(self):
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        self.assertEqual(as_rgb_image(bgr).getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(as_rgb_image(Image.new("RGB", (1, 1), "blue")).getpixel((0, 0)), (0, 0, 255))

    def test_saved_index_is_pickle_free_and_rejects_changed_gallery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gallery = root / "gallery"
            (gallery / "a").mkdir(parents=True)
            image = gallery / "a/reference.png"
            Image.new("RGB", (4, 4), "red").save(image)
            metadata = {"extractor": {"feature_dimension": 2}, "gallery": gallery_manifest(gallery, ["a"])}
            database = VehicleReferenceDB()
            database.metadata = metadata
            database.add("a", np.array([1, 0]), "a/reference.png")
            path = root / "index.npz"
            database.save(path)
            loaded = VehicleReferenceDB.load(path, metadata)
            self.assertEqual(loaded.query(np.array([1, 0]))[0]["label"], "a")
            Image.new("RGB", (4, 4), "blue").save(image)
            changed = {**metadata, "gallery": gallery_manifest(gallery, ["a"])}
            with self.assertRaisesRegex(ValueError, "stale"):
                VehicleReferenceDB.load(path, changed)

    def test_crop_api_does_not_initialize_a_detector(self):
        class Extractor:
            def encode(self, images, batch_size):
                return np.tile([1.0, 0.0], (len(images), 1))
        classifier = DroneVehicleClassifier(device="cpu", use_detector=False, extractor=Extractor())
        classifier.db.add("a", np.array([1, 0]))
        result = classifier.classify_crops([Image.new("RGB", (8, 8))])
        self.assertEqual(result["label"], "a")
        self.assertIsNone(classifier._detector)


if __name__ == "__main__":
    unittest.main()
