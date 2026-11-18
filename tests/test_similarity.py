"""Regression tests for recognition semantics and gallery compatibility."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torchvision import models

from vehicle_recognition.similarity_efficientnet import (
    DroneVehicleClassifier, VehicleReferenceDB, as_rgb_image, gallery_manifest,
    PadToSquare, feature_preprocessing,
)


class SimilarityTests(unittest.TestCase):
    def test_qwen_options_validate_limits_and_propagate_to_classifier(self):
        from vehicle_recognition.configuration import qwen_options, load_settings
        from vehicle_recognition.pipeline import make_classifier

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_root = Path(__file__).resolve().parents[1]
            import yaml
            config = yaml.safe_load((config_root / "configs/yolo26s_qwen_embedding.yaml").read_text())
            config["classes_file"] = str(config_root / "configs/classes.yaml")
            config["datasets_file"] = str(config_root / "configs/datasets.yaml")
            custom = {"min_pixels": 65536, "max_pixels": 262144, "instruction": "Represent vehicle geometry."}
            config["recognition"]["qwen"] = custom
            path = root / "qwen.yaml"
            path.write_text(yaml.safe_dump(config))
            settings = load_settings(root, path)
            settings.config["device"] = "cpu"
            classifier = make_classifier(settings, use_detector=False)
            self.assertEqual(classifier.extractor_options, custom)
            self.assertIsNone(classifier._extractor)
        self.assertEqual(qwen_options()["min_pixels"], 4096)
        for invalid in [{"min_pixels": 262145}, {"min_pixels": True}, {"instruction": " "}, {"pixels": 65536}]:
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                qwen_options(invalid)

    def test_qwen_pooling_uses_last_valid_token_in_variable_length_batch(self):
        from vehicle_recognition.qwen_embeddings import last_token_embedding

        hidden = torch.tensor([[[1., 0.], [0., 2.], [9., 9.]],
                               [[9., 9.], [1., 0.], [3., 4.]]])
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        np.testing.assert_allclose(last_token_embedding(hidden, mask), [[0, 1], [.6, .8]], atol=1e-6)

    def test_square_padding_preserves_all_pixels_including_asymmetric_borders(self):
        for width, height in [(40, 10), (10, 40), (31, 10), (10, 31), (10, 10)]:
            with self.subTest(size=(width, height)):
                pixels = np.arange(width*height*3, dtype=np.uint8).reshape(height, width, 3)
                square = PadToSquare((124, 116, 104))(Image.fromarray(pixels))
                side = max(width, height)
                self.assertEqual(square.size, (side, side))
                left, top = (side-width)//2, (side-height)//2
                np.testing.assert_array_equal(np.asarray(square)[top:top+height, left:left+width], pixels)

    def test_square_preprocessing_retains_end_markers_and_checkpoint_normalization(self):
        checkpoint = models.EfficientNet_B3_Weights.IMAGENET1K_V1
        pixels = np.zeros((100, 300, 3), dtype=np.uint8)
        pixels[:, :20] = (255, 0, 0)
        pixels[:, -20:] = (0, 0, 255)
        image = Image.fromarray(pixels)
        result = feature_preprocessing(checkpoint, "square_pad_resize")(image).numpy()
        self.assertEqual(result.shape, (3, 300, 300))
        preset = checkpoint.transforms()
        restored = result*np.asarray(preset.std)[:, None, None]+np.asarray(preset.mean)[:, None, None]
        np.testing.assert_allclose(restored[:, 150, 0], [1, 0, 0], atol=1e-6)
        np.testing.assert_allclose(restored[:, 150, -1], [0, 0, 1], atol=1e-6)
        np.testing.assert_allclose(result[:, 0, 150], 0, atol=0.01)
        np.testing.assert_array_equal(feature_preprocessing(checkpoint)(image), preset(image))

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
            changed_preprocessing = {**metadata, "extractor": {"feature_dimension": 2,
                                                               "preprocessing_mode": "square_pad_resize"}}
            with self.assertRaisesRegex(ValueError, "stale"):
                VehicleReferenceDB.load(path, changed_preprocessing)
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
