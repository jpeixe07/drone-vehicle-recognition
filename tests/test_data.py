"""Annotation conversion, archive extraction, and seeded augmentation checks."""

from pathlib import Path
import json
import tempfile
import unittest
from zipfile import ZipFile

from PIL import Image

from vehicle_recognition.configuration import ProjectSettings
from vehicle_recognition.datasets import convert_visdrone_split, dataset_names, extract_zip, prepare_weights
from vehicle_recognition.reference_dataset_augmentation import augment_dataset
from vehicle_recognition.similarity_efficientnet import file_sha256


class DataTests(unittest.TestCase):
    def test_conversion_skips_ignored_regions_and_other_classes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "images").mkdir(parents=True)
            (source / "annotations").mkdir()
            Image.new("RGB", (100, 50)).save(source / "images/example.jpg")
            (source / "annotations/example.txt").write_text(
                "10,10,20,10,1,4,0,0\n0,0,100,50,0,0,0,0\n0,0,10,10,1,11,0,0\n")
            names = [f"class_{number}" for number in range(10)]
            result = convert_visdrone_split(source, root / "output", "val", names)
            labels = (root / "output/labels/val/example.txt").read_text().splitlines()
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0], "3 0.20000000 0.30000000 0.20000000 0.20000000")
            self.assertEqual(result["class_counts"]["class_3"], 1)
            self.assertTrue((root / "output/annotations/val/example.txt").is_file())

    def test_archive_cannot_escape_the_dataset_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "bad.zip"
            with ZipFile(archive, "w") as zipped:
                zipped.writestr("../escaped.txt", "data")
            with self.assertRaises(ValueError):
                extract_zip(archive, root / "dataset")
            self.assertFalse((root / "escaped.txt").exists())

    def test_dataset_names_preserve_export_order(self):
        self.assertEqual(dataset_names({"names": {1: "b", 0: "a"}}), ["a", "b"])
        with self.assertRaises(ValueError):
            dataset_names({"names": {1: "a"}})

    def test_changing_detector_source_cannot_silently_reuse_old_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "weights/model.pt"
            target.parent.mkdir()
            target.write_bytes(b"model fixture")
            (target.parent / "manifest.json").write_text(json.dumps({"detectors": {
                "model.pt": {"source": "huggingface", "repo_id": "publisher/old", "requested_revision": "main",
                             "filename": "best.pt", "file": "model.pt", "sha256": file_sha256(target)}
            }}))
            settings = ProjectSettings(root, {"detector": {"weights": "weights/model.pt", "huggingface_repo": "publisher/new",
                                       "huggingface_filename": "best.pt", "huggingface_revision": "main"}}, {}, [])
            with self.assertRaisesRegex(ValueError, "source differs"):
                prepare_weights(settings)

    def test_augmentation_is_repeatable_and_does_not_reaugment_variants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "references/a"
            source.mkdir(parents=True)
            Image.new("RGB", (24, 24), "red").save(source / "original.jpg")
            Image.new("RGB", (24, 24), "blue").save(source / "original_aug_0.jpg")
            first = augment_dataset(source.parent, root / "first", 2, 7, ["a"])
            second = augment_dataset(source.parent, root / "second", 2, 7, ["a"])
            self.assertEqual(first["generated"], 2)
            self.assertEqual(first["images"], 4)
            for path in (root / "first/a").iterdir():
                self.assertEqual(path.read_bytes(), (root / "second/a" / path.name).read_bytes())
            self.assertEqual(first["images"], second["images"])


if __name__ == "__main__":
    unittest.main()
