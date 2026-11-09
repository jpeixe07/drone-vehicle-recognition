"""Offline inference with optional rotated views and explicitly grouped crops."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from .configuration import ProjectSettings, image_paths
from .reporting import RunReporter, new_run_directory
from .similarity_efficientnet import DroneVehicleClassifier, as_rgb_image, file_sha256


def make_classifier(settings: ProjectSettings, use_detector: bool = True) -> DroneVehicleClassifier:
    recognition = settings.config["recognition"]
    return DroneVehicleClassifier(
        yolo_model_path=settings.path(settings.config["detector"]["weights"]),
        backbone=recognition["backbone"], weights=recognition["weights"], device=settings.config["device"],
        use_detector=use_detector, threshold=float(recognition["acceptance_threshold"]),
        top_k=int(recognition["top_k"]), batch_size=int(recognition.get("batch_size", 16)),
        unknown_label=recognition["unknown_label"], detection_settings=settings.config["detector"],
    )


def build_index(settings: ProjectSettings, index_path: Path | None = None) -> dict:
    classifier = make_classifier(settings, use_detector=False)
    metadata = classifier.build_reference_db(settings.gallery, settings.labels)
    path = index_path or settings.index
    classifier.db.save(path)
    return {"index": str(path), "references": len(classifier.db.entries), "gallery_sha256": metadata["gallery"]["sha256"],
            "extractor": metadata["extractor"]}


def bbox_iou(first: list, second: list) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def map_box_to_original(box: list, matrix: np.ndarray, width: int, height: int) -> list[float]:
    corners = np.asarray([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], dtype=np.float64)
    mapped = cv2.transform(corners[None], cv2.invertAffineTransform(matrix))[0]
    return [float(np.clip(mapped[:, 0].min(), 0, width)), float(np.clip(mapped[:, 1].min(), 0, height)),
            float(np.clip(mapped[:, 0].max(), 0, width)), float(np.clip(mapped[:, 1].max(), 0, height))]


def recognize_image(classifier: DroneVehicleClassifier, image: Image.Image, angles: list,
                    duplicate_iou: float = 0.5) -> list[dict]:
    if not angles or not all(math.isfinite(float(angle)) for angle in angles):
        raise ValueError("At least one finite rotation angle is required")
    if not 0 < duplicate_iou <= 1:
        raise ValueError("Rotation duplicate IoU must be between 0 and 1")
    angles = list(dict.fromkeys(float(angle) % 360 for angle in angles))
    width, height = image.size
    candidates = []
    for angle in angles:
        pixels, matrix = classifier.rotate_image_bound(np.asarray(image), angle, return_matrix=True)
        for detection in classifier.detect_vehicle_crops(Image.fromarray(pixels)):
            mapped = map_box_to_original(detection["bbox"], matrix, width, height)
            if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
                continue
            candidates.append({**detection, "bbox": mapped, "rotation": angle})
    groups = []
    for candidate in sorted(candidates, key=lambda item: -item["detection_confidence"]):
        compatible = [(bbox_iou(candidate["bbox"], group[0]["bbox"]), group) for group in groups
                      if all(item["rotation"] != candidate["rotation"] for item in group)]
        best = max(compatible, key=lambda value: value[0]) if compatible else None
        if best is not None and best[0] >= duplicate_iou:
            best[1].append(candidate)
        else:
            groups.append([candidate])
    results = []
    for group in groups:
        anchor = group[0]
        recognized = classifier.classify_crops([item["_crop"] for item in group])
        results.append({"bbox": anchor["bbox"], "detector_class": anchor["detector_class"],
                        "detection_confidence": anchor["detection_confidence"],
                        "rotations_degrees": [item["rotation"] for item in group], **recognized})
    return results


def run_predictions(settings: ProjectSettings, source: Path | None = None, crops: bool = False,
                    groups_path: Path | None = None, index_path: Path | None = None,
                    output: Path | None = None) -> dict:
    if (source is None) == (groups_path is None):
        raise ValueError("Provide either an image input or an explicit crop-group manifest")
    paths = image_paths(source) if source is not None else []
    groups = None
    if groups_path is not None:
        groups = json.loads(groups_path.read_text(encoding="utf-8"))
        if not isinstance(groups, dict) or not groups:
            raise ValueError("Crop-group manifest must map group names to nonempty image-path lists")
        for group, names in groups.items():
            if not group or not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
                raise ValueError(f"Invalid crop group: {group}")
            for name in names:
                if not (groups_path.parent / name).is_file():
                    raise FileNotFoundError(f"Crop-group image does not exist: {name}")
    classifier = make_classifier(settings, use_detector=not (crops or groups_path is not None))
    path = index_path or settings.index
    if not path.is_file():
        raise FileNotFoundError(f"Reference index does not exist: {path}. Run 'index' first.")
    classifier.load_reference_db(path, settings.gallery, settings.labels)
    metadata = {"mode": "grouped_crops" if groups_path else "crops" if crops else "detection_and_recognition",
                "configuration": settings.config, "labels": settings.labels, "device": classifier.device,
                "reference_index": classifier.db.metadata, "reference_index_sha256": file_sha256(path)}
    if not (crops or groups_path):
        detector_path = settings.path(settings.config["detector"]["weights"])
        if not detector_path.is_file():
            raise FileNotFoundError("Detector weights are missing. Run 'prepare' first.")
        metadata["detector"] = classifier.detector_metadata()
    directory = new_run_directory(settings.path(settings.config["inference"]["output_directory"]), output)
    reporter = RunReporter(directory, metadata, settings.config["inference"].get("save_annotated_images", True))
    started = time.perf_counter()
    if groups_path is not None:
        for group, names in sorted(groups.items()):
            observations = [(groups_path.parent / name).resolve() for name in names]
            before = time.perf_counter()
            recognized = classifier.classify_crops(observations)
            reporter.record({"image": str(group), "observations": names,
                             "observation_sha256": [file_sha256(path) for path in observations],
                             "width": None, "height": None,
                             "elapsed_seconds": time.perf_counter() - before,
                             "detections": [{"bbox": None, "detector_class": None,
                                             "detection_confidence": None, **recognized}]})
    else:
        base = source if source.is_dir() else source.parent
        for path in paths:
            before = time.perf_counter()
            identifier = path.relative_to(base).as_posix()
            try:
                image = as_rgb_image(path)
            except (OSError, ValueError) as error:
                reporter.record({"image": identifier, "error": str(error), "detections": []})
                continue
            if crops:
                detections = [{"bbox": [0, 0, image.width, image.height], "detector_class": None,
                               "detection_confidence": None, **classifier.classify_crops([image])}]
            else:
                inference = settings.config["inference"]
                detections = recognize_image(classifier, image, inference["rotations_degrees"],
                                             float(inference.get("duplicate_iou_threshold", 0.5)))
            reporter.record({"image": identifier, "image_sha256": file_sha256(path),
                             "width": image.width, "height": image.height, "detections": detections,
                             "elapsed_seconds": time.perf_counter() - before}, image)
    return reporter.finish(time.perf_counter() - started)


def collect_crops(settings: ProjectSettings, source: Path, output: Path | None = None) -> dict:
    """Export unlabeled vehicle candidates for manual reference selection."""
    classifier = make_classifier(settings)
    paths = image_paths(source)
    detector_identity = classifier.detector_metadata()
    base = source if source.is_dir() else source.parent
    directory = new_run_directory(settings.path(settings.config["inference"]["output_directory"]), output)
    crop_directory = directory / "crops"
    crop_directory.mkdir()
    reporter = RunReporter(directory, {"mode": "candidate_collection", "configuration": settings.config,
                                      "device": classifier.device, "detector": detector_identity}, save_images=False)
    started = time.perf_counter()
    for path in paths:
        image = as_rgb_image(path)
        identifier = path.relative_to(base).as_posix()
        detections = classifier.detect_vehicle_crops(image)
        for number, detection in enumerate(detections):
            crop = detection.pop("_crop")
            digest = hashlib.sha256(identifier.encode()).hexdigest()[:8]
            name = f"{path.stem}_{digest}_{number:03}.jpg"
            crop.save(crop_directory / name, quality=95)
            detection["crop_file"] = f"crops/{name}"
        reporter.record({"image": identifier, "width": image.width, "height": image.height, "detections": detections})
    return reporter.finish(time.perf_counter() - started)
