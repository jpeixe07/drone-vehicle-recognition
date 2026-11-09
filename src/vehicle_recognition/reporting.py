"""Persistent per-image predictions, experiment metadata, and annotated images."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import uuid

from PIL import Image, ImageDraw


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def new_run_directory(output_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        directory = Path(explicit)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"Output directory is not empty: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = output_root / f"{timestamp}_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def runtime_versions() -> dict:
    return {name: version(name) for name in ["torch", "torchvision", "ultralytics", "numpy", "Pillow", "opencv-python"]}


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class RunReporter:
    """Keep empty detections and rejected matches, including during partial runs."""

    fields = ["image", "x1", "y1", "x2", "y2", "crop_file", "detector_class", "detection_confidence",
              "label", "candidate_label", "similarity", "accepted", "observation_count", "top_candidates"]

    def __init__(self, directory: Path, metadata: dict, save_images: bool = True):
        self.directory = directory
        self.metadata = {"schema_version": 1, **metadata, "package_versions": runtime_versions(),
                         "source_sha256": source_fingerprint()}
        self.save_images = save_images
        self.records = []
        write_json(directory / "run.json", self.metadata)
        with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=self.fields).writeheader()

    def record(self, record: dict, image: Image.Image | None = None) -> None:
        self.records.append(record)
        with (self.directory / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        with (self.directory / "predictions.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fields)
            for prediction in record["detections"]:
                box = prediction.get("bbox") or [None] * 4
                row = {name: prediction.get(name) for name in self.fields if name not in {"image", "x1", "y1", "x2", "y2"}}
                row.update(dict(zip(["x1", "y1", "x2", "y2"], box)))
                row["image"] = record["image"]
                row["top_candidates"] = json.dumps(prediction.get("top_candidates", []), ensure_ascii=False)
                writer.writerow(row)
        if image is not None and self.save_images:
            self._annotate(record, image)

    def _annotate(self, record: dict, image: Image.Image) -> None:
        annotated = image.copy().convert("RGB")
        drawing = ImageDraw.Draw(annotated)
        for prediction in record["detections"]:
            if prediction.get("bbox") is None:
                continue
            box = prediction["bbox"]
            color = "#21a657" if prediction.get("accepted") else "#e79b24"
            label = prediction.get("label") or prediction.get("detector_class", "vehicle")
            if prediction.get("similarity") is not None:
                label += f" sim={prediction['similarity']:.3f}"
            if prediction.get("detection_confidence") is not None:
                label += f" det={prediction['detection_confidence']:.3f}"
            drawing.rectangle(box, outline=color, width=3)
            x, y = max(0, box[0]), max(0, box[1] - 14)
            text_box = drawing.textbbox((x, y), label)
            drawing.rectangle(text_box, fill=color)
            drawing.text((x, y), label, fill="black")
        identifier = hashlib.sha256(record["image"].encode()).hexdigest()[:10]
        directory = self.directory / "annotated"
        directory.mkdir(exist_ok=True)
        annotated.save(directory / f"{Path(record['image']).stem}_{identifier}.jpg", quality=95)

    def finish(self, elapsed_seconds: float) -> dict:
        failed = sum(bool(record.get("error")) for record in self.records)
        summary = {
            "output_directory": str(self.directory), "images": len(self.records), "failed_images": failed,
            "detections": sum(len(record["detections"]) for record in self.records),
            "accepted": sum(bool(item.get("accepted")) for record in self.records for item in record["detections"]),
            "elapsed_seconds": elapsed_seconds,
        }
        write_json(self.directory / "predictions.json", {"schema_version": 1, "images": self.records})
        write_json(self.directory / "summary.json", summary)
        return summary
