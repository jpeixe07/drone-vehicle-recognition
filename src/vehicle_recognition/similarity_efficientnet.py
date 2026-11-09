"""Pretrained visual features, reference matching, and YOLO vehicle crops."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models

from .configuration import IMAGE_SUFFIXES


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_rgb_image(image: Image.Image | np.ndarray | str | Path) -> Image.Image:
    """Paths/PIL images are RGB; NumPy frames follow OpenCV's BGR convention."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB")
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError("OpenCV image arrays must use uint8 pixels")
    if array.ndim == 2:
        return Image.fromarray(array).convert("RGB")
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("Expected a grayscale, BGR, or BGRA image")
    conversion = cv2.COLOR_BGR2RGB if array.shape[2] == 3 else cv2.COLOR_BGRA2RGB
    return Image.fromarray(cv2.cvtColor(array, conversion))


def resolve_device(device: str = "auto") -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be auto, cpu, or cuda")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def backbone_spec(backbone: str, weights: str = "IMAGENET1K_V1") -> tuple:
    registry = {
        "efficientnet_b3": (models.efficientnet_b3, models.EfficientNet_B3_Weights, 1536),
        "convnext_base": (models.convnext_base, models.ConvNeXt_Base_Weights, 1024),
        "resnet50": (models.resnet50, models.ResNet50_Weights, 2048),
    }
    if backbone not in registry:
        raise ValueError(f"Unsupported feature backbone: {backbone}")
    builder, enum, dimension = registry[backbone]
    try:
        checkpoint = enum[weights]
    except KeyError as error:
        raise ValueError(f"Unsupported weights {weights!r} for {backbone}") from error
    return builder, checkpoint, dimension


def checkpoint_path(checkpoint) -> Path:
    return Path(torch.hub.get_dir()) / "checkpoints" / Path(urlparse(checkpoint.url).path).name


def checkpoint_sha256(checkpoint) -> str:
    path = checkpoint_path(checkpoint)
    digest = file_sha256(path)
    expected = re.search(r"-([a-f0-9]{8,64})\.", path.name)
    if expected and not digest.startswith(expected.group(1)):
        raise ValueError(f"Checkpoint checksum mismatch: {path.name}. Remove it and run 'prepare' again.")
    return digest


class VehicleFeatureExtractor(nn.Module):
    """Frozen pretrained backbone with pooled, L2-normalized visual features."""

    def __init__(self, backbone: str = "efficientnet_b3", weights: str = "IMAGENET1K_V1",
                 device: str = "auto", allow_download: bool = False):
        super().__init__()
        builder, checkpoint, self.feature_dim = backbone_spec(backbone, weights)
        if not checkpoint_path(checkpoint).is_file() and not allow_download:
            raise FileNotFoundError("Feature weights are not cached. Run 'prepare' before indexing or prediction.")
        if allow_download:
            checkpoint.get_state_dict(progress=True, check_hash=True)
        digest = checkpoint_sha256(checkpoint)
        base = builder(weights=checkpoint)
        self.encoder = nn.Sequential(*list(base.children())[:-2]) if backbone == "resnet50" else base.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.transform = checkpoint.transforms()
        self.device = resolve_device(device)
        self.to(self.device).eval().requires_grad_(False)
        self.metadata = {
            "backbone": backbone, "weights": checkpoint.name, "checkpoint_sha256": digest,
            "feature_dimension": self.feature_dim, "preprocessing": repr(self.transform),
            "normalization": "l2", "projection_head": False, "torchvision": torchvision.__version__.split("+")[0],
            "implementation_sha256": file_sha256(Path(__file__)),
        }

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.pooling(self.encoder(batch)).flatten(1), p=2, dim=1)

    def encode(self, images: list, batch_size: int = 16) -> np.ndarray:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if not images:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        chunks = []
        self.eval()
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = torch.stack([self.transform(as_rgb_image(image)) for image in images[start:start + batch_size]])
                chunks.append(self(batch.to(self.device)).cpu().numpy())
        return np.concatenate(chunks).astype(np.float32, copy=False)


def gallery_manifest(directory: Path, labels: list[str] | None = None) -> dict:
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Reference gallery does not exist: {directory}")
    available = sorted(path.name for path in directory.iterdir() if path.is_dir())
    labels = list(labels) if labels is not None else available
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Reference labels must be nonempty and unique")
    unexpected = set(available) - set(labels)
    if unexpected:
        raise ValueError(f"Unconfigured reference classes: {', '.join(sorted(unexpected))}")
    entries = []
    for label in labels:
        if Path(label).name != label or label in {".", ".."}:
            raise ValueError("Reference labels must be single directory names")
        paths = sorted(path for path in (directory / label).rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not paths:
            raise ValueError(f"Reference class has no images: {label}")
        entries.extend({"label": label, "path": path.relative_to(directory).as_posix(), "sha256": file_sha256(path)}
                       for path in paths)
    serialized = json.dumps({"labels": labels, "entries": entries}, sort_keys=True).encode()
    return {"labels": labels, "entries": entries, "sha256": hashlib.sha256(serialized).hexdigest()}


def normalized_vector(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1 or not value.size or not np.isfinite(value).all():
        raise ValueError("Embeddings must be finite, nonempty vectors")
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("A zero embedding cannot be compared by cosine similarity")
    return value / norm


@dataclass
class ReferenceEntry:
    label: str
    embedding: np.ndarray
    reference: str = ""


class VehicleReferenceDB:
    """One vector per reference, with the maximum cosine similarity per class."""

    def __init__(self, threshold: float = 0.78, unknown_label: str = "unknown"):
        if not -1 <= threshold <= 1:
            raise ValueError("Cosine threshold must be between -1 and 1")
        self.threshold = threshold
        self.unknown_label = unknown_label
        self.entries: list[ReferenceEntry] = []
        self.metadata: dict = {}
        self._matrix = None

    def add(self, label: str, embedding: np.ndarray, reference: str = "") -> None:
        vector = normalized_vector(embedding)
        if not label or label == self.unknown_label:
            raise ValueError("References need a known model label")
        if self.entries and vector.shape != self.entries[0].embedding.shape:
            raise ValueError("Reference embeddings have incompatible dimensions")
        self.entries.append(ReferenceEntry(label, vector, reference))
        self._matrix = None

    def query(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        return self.aggregate(normalized_vector(query_embedding)[None], top_k)["top_candidates"]

    def aggregate(self, embeddings: np.ndarray, top_k: int = 3) -> dict:
        """Average complete per-class scores across explicitly associated observations."""
        if not self.entries:
            raise ValueError("Reference gallery is empty; build or load an index first")
        values = np.asarray(embeddings)
        if values.ndim != 2 or not len(values):
            raise ValueError("At least one observation embedding is required")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        queries = np.stack([normalized_vector(value) for value in values])
        if self._matrix is None:
            self._matrix = np.stack([entry.embedding for entry in self.entries])
        if queries.shape[1] != self._matrix.shape[1]:
            raise ValueError("Query and reference embeddings have incompatible dimensions")
        similarities = np.clip(queries @ self._matrix.T, -1.0, 1.0)
        candidates = []
        for label in sorted({entry.label for entry in self.entries}):
            indices = [i for i, entry in enumerate(self.entries) if entry.label == label]
            per_reference = similarities[:, indices]
            score = float(per_reference.max(axis=1).mean())
            best_reference = indices[int(np.unravel_index(per_reference.argmax(), per_reference.shape)[1])]
            candidates.append({"label": label, "similarity": score, "match": score >= self.threshold,
                               "reference": self.entries[best_reference].reference})
        candidates.sort(key=lambda item: (-item["similarity"], item["label"]))
        best = candidates[0]
        return {
            "label": best["label"] if best["match"] else self.unknown_label,
            "candidate_label": best["label"], "similarity": best["similarity"], "accepted": best["match"],
            "observation_count": len(queries), "aggregation": "mean_class_maximum_similarity",
            "top_candidates": candidates[:top_k],
        }

    def save(self, path: Path) -> None:
        if not self.entries:
            raise ValueError("Cannot save an empty reference index")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, embeddings=np.stack([entry.embedding for entry in self.entries]),
                                    labels=np.asarray([entry.label for entry in self.entries]),
                                    references=np.asarray([entry.reference for entry in self.entries]),
                                    metadata=np.asarray(json.dumps(self.metadata, sort_keys=True)))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, expected_metadata: dict, threshold: float = 0.78,
             unknown_label: str = "unknown") -> "VehicleReferenceDB":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata != expected_metadata:
                raise ValueError("Reference index is incompatible or stale. Rebuild it with 'index'.")
            embeddings, labels, references = archive["embeddings"], archive["labels"], archive["references"]
            if embeddings.ndim != 2 or not len(embeddings) or len(labels) != len(embeddings) or len(references) != len(labels):
                raise ValueError("Malformed reference index arrays")
            dimension = metadata.get("extractor", {}).get("feature_dimension")
            if dimension is not None and embeddings.shape[1] != dimension:
                raise ValueError("Reference index feature dimension does not match its metadata")
            expected_entries = metadata.get("gallery", {}).get("entries")
            if expected_entries is not None:
                if list(labels) != [entry["label"] for entry in expected_entries] or list(references) != [entry["path"] for entry in expected_entries]:
                    raise ValueError("Reference index labels or paths do not match its gallery metadata")
            database = cls(threshold, unknown_label)
            database.metadata = metadata
            for label, embedding, reference in zip(labels, embeddings, references):
                database.add(str(label), embedding, str(reference))
        return database


class DroneVehicleClassifier:
    """Detector-independent crop recognition, with optional YOLO localization."""

    def __init__(self, yolo_model_path: str | Path | None = None, backbone: str = "efficientnet_b3",
                 device: str = "auto", weights: str = "IMAGENET1K_V1", use_detector: bool = True,
                 threshold: float = 0.78, top_k: int = 3, batch_size: int = 16,
                 unknown_label: str = "unknown", detection_settings: dict | None = None,
                 extractor=None, detector=None):
        self.device = resolve_device(device)
        self.yolo_model_path = Path(yolo_model_path) if yolo_model_path else None
        self.backbone, self.weights = backbone, weights
        self.use_detector = use_detector
        self.top_k, self.batch_size = top_k, batch_size
        self.detection_settings = detection_settings or {}
        self._extractor, self._detector = extractor, detector
        self.db = VehicleReferenceDB(threshold, unknown_label)

    @property
    def extractor(self):
        if self._extractor is None:
            self._extractor = VehicleFeatureExtractor(self.backbone, self.weights, self.device)
        return self._extractor

    @property
    def detector(self):
        if not self.use_detector:
            raise ValueError("This classifier is configured for crop recognition without a detector")
        if self._detector is None:
            if self.yolo_model_path is None or not self.yolo_model_path.is_file():
                raise FileNotFoundError("Detector weights do not exist. Run 'prepare' or configure a local weights file.")
            from ultralytics import YOLO

            self._detector = YOLO(str(self.yolo_model_path))
        return self._detector

    def index_metadata(self, directory: Path, allowed_labels: list[str] | None = None) -> dict:
        return {"schema_version": 1, "extractor": self.extractor.metadata,
                "gallery": gallery_manifest(directory, allowed_labels)}

    def build_reference_db(self, reference_dir: str | Path, allowed_labels: list[str] | None = None) -> dict:
        directory = Path(reference_dir)
        metadata = self.index_metadata(directory, allowed_labels)
        entries = metadata["gallery"]["entries"]
        embeddings = self.extractor.encode([directory / entry["path"] for entry in entries], self.batch_size)
        if len(embeddings) != len(entries):
            raise ValueError("The extractor did not return one embedding per reference image")
        database = VehicleReferenceDB(self.db.threshold, self.db.unknown_label)
        database.metadata = metadata
        for entry, embedding in zip(entries, embeddings):
            database.add(entry["label"], embedding, entry["path"])
        self.db = database
        return metadata

    def load_reference_db(self, index_path: Path, reference_dir: Path, allowed_labels: list[str] | None = None) -> None:
        metadata = self.index_metadata(reference_dir, allowed_labels)
        self.db = VehicleReferenceDB.load(index_path, metadata, self.db.threshold, self.db.unknown_label)

    def classify_crop(self, crop) -> list[dict]:
        return self.classify_crops([crop])["top_candidates"]

    def classify_crops(self, crops: list) -> dict:
        if not crops:
            raise ValueError("No crops were supplied for recognition")
        if not self.db.entries:
            raise ValueError("Reference gallery is empty; build or load an index first")
        return self.db.aggregate(self.extractor.encode(crops, self.batch_size), self.top_k)

    def detect_vehicle_crops(self, image) -> list[dict]:
        image = as_rgb_image(image)
        settings = self.detection_settings
        raw_names = self.detector.names
        names = dict(enumerate(raw_names)) if isinstance(raw_names, list) else {int(key): value for key, value in raw_names.items()}
        allowed = {str(name).strip().casefold() for name in settings.get("vehicle_classes", ["car", "van", "truck", "bus"])}
        class_ids = [int(key) for key, name in names.items() if str(name).strip().casefold() in allowed]
        if not class_ids:
            raise ValueError("The detector has none of the configured vehicle class names")
        prediction = self.detector.predict(
            source=image, classes=class_ids, conf=float(settings.get("confidence_threshold", 0.4)),
            iou=float(settings.get("iou_threshold", 0.5)), imgsz=int(settings.get("image_size", 1280)),
            agnostic_nms=True, device=0 if self.device == "cuda" else "cpu", verbose=False,
        )[0]
        if prediction.boxes is None:
            raise ValueError("This pipeline requires a detector with horizontal bounding boxes")
        width, height = image.size
        margin = int(settings.get("crop_margin_pixels", 10))
        detections = []
        for box in prediction.boxes:
            class_id = int(box.cls.item())
            if class_id not in class_ids:
                continue
            coordinates = box.xyxy[0].detach().cpu().numpy().astype(float)
            if not np.isfinite(coordinates).all():
                raise ValueError("Detector produced a non-finite bounding box")
            x1, y1, x2, y2 = coordinates.tolist()
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop_box = (max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin)),
                        min(width, math.ceil(x2 + margin)), min(height, math.ceil(y2 + margin)))
            detections.append({"bbox": [x1, y1, x2, y2], "crop_bbox": list(crop_box),
                               "detector_class": str(names[class_id]), "detection_confidence": float(box.conf.item()),
                               "_crop": image.crop(crop_box)})
        return detections

    def classify_frame(self, frame) -> list[dict]:
        results = []
        for detection in self.detect_vehicle_crops(frame):
            crop = detection.pop("_crop")
            results.append({**detection, **self.classify_crops([crop])})
        return results

    def detector_metadata(self) -> dict:
        path = self.yolo_model_path
        if path is None or not path.is_file():
            raise FileNotFoundError("Detector weights do not exist. Run 'prepare' first.")
        identity = {"source": "local", "file": path.name, "sha256": file_sha256(path)}
        manifest_path = path.parent / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            entry = manifest.get("detectors", {}).get(path.name) or manifest.get("detector", {})
            if entry.get("sha256") == identity["sha256"]:
                allowed = {"source", "file", "sha256", "repo_id", "revision", "requested_revision", "filename"}
                identity.update({key: value for key, value in entry.items() if key in allowed})
        return identity

    @staticmethod
    def rotate_image_bound(image: np.ndarray, angle: float, return_matrix: bool = False):
        height, width = image.shape[:2]
        angle = float(angle) % 360
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_width = max(1, math.ceil(width * cosine + height * sine - 1e-9))
        new_height = max(1, math.ceil(height * cosine + width * sine - 1e-9))
        matrix[0, 2] += (new_width - width) / 2
        matrix[1, 2] += (new_height - height) / 2
        rotated = cv2.warpAffine(image, matrix, (new_width, new_height))
        return (rotated, matrix) if return_matrix else rotated
