"""Final annotated evaluation with frozen references and decision thresholds."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .configuration import ProjectSettings, image_paths, read_yaml
from .datasets import dataset_names
from .pipeline import bbox_iou, make_classifier, recognize_image
from .reporting import RunReporter, new_run_directory, write_json
from .similarity_efficientnet import as_rgb_image, file_sha256


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def precision_recall(correct: int, predicted: int, actual: int) -> dict:
    return {"true_positives": correct, "false_positives": predicted - correct,
            "false_negatives": actual - correct, "precision": ratio(correct, predicted),
            "recall": ratio(correct, actual), "f1": ratio(2 * correct, predicted + actual)}


class RecognitionScores:
    def __init__(self, labels: list[str], unknown: str):
        self.labels = labels + [unknown]
        self.unknown = unknown
        self.matrix = {label: Counter() for label in self.labels}
        self.counts = Counter({key: 0 for key in ("objects", "correct", "accepted", "accepted_correct", "known_objects",
                                                "known_top1_correct", "known_topk_correct", "unknown_objects", "unknown_rejected")})
        self.per_class = {label: Counter() for label in self.labels}

    def add(self, truth: str, prediction: dict) -> None:
        self.matrix[truth][prediction["label"]] += 1
        known = truth != self.unknown
        observations = {"objects": 1, "correct": int(prediction["label"] == truth),
                        "accepted": int(prediction["accepted"]),
                        "accepted_correct": int(prediction["accepted"] and prediction["label"] == truth),
                        "known_objects": int(known),
                        "known_top1_correct": int(known and prediction["candidate_label"] == truth),
                        "known_topk_correct": int(known and truth in [p["label"] for p in prediction["top_candidates"]]),
                        "unknown_objects": int(not known),
                        "unknown_rejected": int(not known and not prediction["accepted"])}
        self.counts.update(observations)
        self.per_class[truth].update(observations)

    def result(self) -> dict:
        counts = self.counts
        per_class = {label: {"objects": values["objects"], "correct": values["correct"],
                             "accuracy": ratio(values["correct"], values["objects"]),
                             "rejected": values["objects"] - values["accepted"]}
                     for label, values in self.per_class.items()}
        supported = [values["accuracy"] for values in per_class.values() if values["objects"]]
        return {**dict(counts), "accuracy": ratio(counts["correct"], counts["objects"]),
                "macro_accuracy": sum(supported) / len(supported) if supported else None,
                "raw_top1_accuracy_known": ratio(counts["known_top1_correct"], counts["known_objects"]),
                "raw_topk_accuracy_known": ratio(counts["known_topk_correct"], counts["known_objects"]),
                "accepted_accuracy": ratio(counts["accepted_correct"], counts["accepted"]),
                "coverage": ratio(counts["accepted"], counts["objects"]),
                "unknown_rejection_recall": ratio(counts["unknown_rejected"], counts["unknown_objects"]),
                "per_class": per_class,
                "confusion_matrix": {truth: {label: self.matrix[truth][label] for label in self.labels}
                                     for truth in self.labels}}


def evaluation_samples(dataset_yaml: Path, requested: list[str]) -> tuple[list[str], list[dict]]:
    dataset = read_yaml(dataset_yaml)
    names = dataset_names(dataset)
    base = Path(dataset.get("path") or dataset_yaml.parent)
    previous_root = base
    if not base.is_absolute():
        base = dataset_yaml.parent / base
    elif not base.is_dir():
        # Accept legacy exports mounted at a different root, including read-only
        # datasets whose YAML still contains the original host's absolute path.
        base = dataset_yaml.parent
    splits = [split for split in ("train", "val", "test") if dataset.get(split)] if requested == ["all"] else requested
    if not splits or "all" in splits:
        raise ValueError("Choose 'all' alone or explicit train/val/test splits")
    samples, seen = [], set()
    for split in dict.fromkeys(splits):
        sources = dataset.get(split)
        if not sources:
            raise ValueError(f"Dataset does not define split {split}")
        for source in sources if isinstance(sources, list) else [sources]:
            relative = Path(source)
            if (relative.is_absolute() and not relative.exists() and previous_root.is_absolute()
                    and relative.is_relative_to(previous_root)):
                relative = relative.relative_to(previous_root)
            source = relative if relative.is_absolute() else base / relative
            adjacent = dataset_yaml.parent / relative
            if not relative.is_absolute() and adjacent.exists():
                source = adjacent
            if not source.exists() and relative.parts[:1] == ("..",):
                source = base / Path(*relative.parts[1:])
            if not source.exists() and requested == ["all"]:
                print(f"Export has no directory for split {split}; evaluating all existing splits", file=sys.stderr)
                continue
            if source.suffix.lower() == ".txt" and source.is_file():
                paths = []
                for row in source.read_text().splitlines():
                    if row.strip():
                        path = Path(row.strip())
                        paths.append(path if path.is_absolute() else source.parent / path)
            else:
                paths = image_paths(source)
            for path in paths:
                path = path.resolve()
                if path in seen:
                    raise ValueError(f"Image listed more than once in selected splits: {path}")
                seen.add(path)
                parts = list(path.parts)
                if "images" not in parts:
                    raise ValueError(f"Expected a YOLO images directory: {path}")
                position = len(parts) - 1 - parts[::-1].index("images")
                parts[position] = "labels"
                annotation = Path(*parts).with_suffix(".txt")
                if not annotation.is_file():
                    raise FileNotFoundError(f"Annotation missing (use an empty file for negatives): {annotation}")
                samples.append({"split": split, "path": path, "annotation": annotation,
                                "identifier": path.relative_to(dataset_yaml.parent).as_posix()})
    if not samples:
        raise ValueError("No evaluation images were found")
    return names, samples


def annotations(path: Path, names: list[str], width: int, height: int) -> list[dict]:
    objects = []
    for number, row in enumerate(path.read_text().splitlines(), 1):
        if not row.strip():
            continue
        fields = row.split()
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO detection annotation with 5 fields: {path}:{number}")
        class_id = int(fields[0])
        x, y, w, h = map(float, fields[1:])
        if not (0 <= class_id < len(names) and all(math.isfinite(v) for v in (x, y, w, h))
                and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"Invalid class ID or normalized bounding box: {path}:{number}")
        box = [max(0.0, (x - w / 2) * width), max(0.0, (y - h / 2) * height),
               min(float(width), (x + w / 2) * width), min(float(height), (y + h / 2) * height)]
        objects.append({"label": names[class_id], "bbox": box})
    return objects


def match_detections(truth: list[dict], predictions: list[dict], iou_threshold: float) -> list[tuple[int, int, float]]:
    """Class-agnostic one-to-one matches, ordered by detector confidence."""
    remaining = set(range(len(truth)))
    matches = []
    for predicted in sorted(range(len(predictions)), key=lambda i: -predictions[i]["detection_confidence"]):
        candidates = [(bbox_iou(predictions[predicted]["bbox"], truth[actual]["bbox"]), actual)
                      for actual in sorted(remaining)]
        if candidates:
            overlap, actual = max(candidates, key=lambda item: item[0])
            if overlap >= iou_threshold:
                matches.append((actual, predicted, overlap))
                remaining.remove(actual)
    return matches


def run_evaluation(settings: ProjectSettings, splits: list[str] | None = None,
                   iou_threshold: float = 0.5, output: Path | None = None) -> dict:
    if not 0 < iou_threshold <= 1:
        raise ValueError("Evaluation IoU must be in (0, 1]")
    dataset_yaml = settings.path(settings.datasets["forest_vehicles"]["directory"]) / "data.yaml"
    names, samples = evaluation_samples(dataset_yaml, splits or ["all"])
    unknown = settings.config["recognition"]["unknown_label"]
    if set(names) - set(settings.labels) - {unknown}:
        raise ValueError("Dataset labels do not match the reference catalog")
    classifier = make_classifier(settings)
    classifier.load_reference_db(settings.index, settings.gallery, settings.labels)
    gallery_hashes = {entry["sha256"] for entry in classifier.db.metadata["gallery"]["entries"]}
    inventory, digests = [], Counter()
    for sample in samples:
        digest = file_sha256(sample["path"])
        if digest in gallery_hashes:
            raise ValueError(f"Evaluation image duplicates a gallery reference: {sample['identifier']}")
        digests[digest] += 1
        inventory.append({"image": sample["identifier"], "split": sample["split"],
                          "image_sha256": digest, "annotation_sha256": file_sha256(sample["annotation"])})
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    directory = new_run_directory(settings.path(settings.config["inference"]["output_directory"]), output)
    metadata = {"mode": "final_evaluation", "configuration": settings.config, "device": classifier.device,
                "labels": names, "reference_index": classifier.db.metadata,
                "reference_index_sha256": file_sha256(settings.index), "detector": classifier.detector_metadata(),
                "dataset_sha256": fingerprint, "dataset_yaml_sha256": file_sha256(dataset_yaml),
                "splits": list(dict.fromkeys(sample["split"] for sample in samples)),
                "protocol": {"training": False, "threshold_tuning": False, "iou_threshold": iou_threshold,
                             "matching": "class_agnostic_greedy_descending_detector_confidence",
                             "gt_crops": "annotation_boxes_without_detector_margin",
                             "duplicates": "every_exported_image_evaluated; exact_duplicate_count_reported"}}
    manifest = dataset_yaml.parent / "manifest.json"
    metadata["dataset_source"] = json.loads(manifest.read_text()) if manifest.is_file() else None
    reporter = RunReporter(directory, metadata, settings.config["inference"].get("save_annotated_images", True))
    crop_directory = directory / "gt_crops"
    crop_directory.mkdir()
    crop_reporter = RunReporter(crop_directory, {**metadata, "mode": "ground_truth_crop_evaluation"}, save_images=False)
    write_json(directory / "dataset_inventory.json", {"images": inventory, "sha256": fingerprint})
    crop_scores = RecognitionScores(settings.labels, unknown)
    matched_scores = RecognitionScores(settings.labels, unknown)
    counts, class_counts, split_counts = Counter(), {label: Counter() for label in names}, Counter()
    started = time.perf_counter()
    with (directory / "objects.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["image", "split", "true_label", "status", "predicted_label", "candidate_label", "similarity", "iou"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for number, (sample, identity) in enumerate(zip(samples, inventory), 1):
            before = time.perf_counter()
            image = as_rgb_image(sample["path"])
            truth = annotations(sample["annotation"], names, image.width, image.height)
            inference = settings.config["inference"]
            predictions = recognize_image(classifier, image, inference["rotations_degrees"],
                                          float(inference.get("duplicate_iou_threshold", 0.5)))
            crop_predictions = []
            for actual in truth:
                prediction = classifier.classify_crops([image.crop(actual["bbox"])])
                crop_scores.add(actual["label"], prediction)
                crop_predictions.append({"bbox": actual["bbox"], "true_label": actual["label"], **prediction})
            matches = match_detections(truth, predictions, iou_threshold)
            by_truth = {actual: (predicted, overlap) for actual, predicted, overlap in matches}
            matched_predictions = {predicted for _, predicted, _ in matches}
            counts.update(images=1, ground_truth=len(truth), predictions=len(predictions), matched=len(matches),
                          positive_images=int(bool(truth)), negative_images=int(not truth))
            split_counts[sample["split"]] += 1
            for actual_index, actual in enumerate(truth):
                label = actual["label"]
                class_counts[label]["objects"] += 1
                row = {"image": sample["identifier"], "split": sample["split"], "true_label": label, "status": "missed"}
                if actual_index in by_truth:
                    predicted_index, overlap = by_truth[actual_index]
                    prediction = predictions[predicted_index]
                    matched_scores.add(label, prediction)
                    correct = prediction["label"] == label
                    counts["end_to_end_correct"] += int(correct)
                    class_counts[label].update(detected=1, correct=int(correct))
                    row.update(status="correct" if correct else "misclassified_or_rejected", iou=overlap,
                               predicted_label=prediction["label"], candidate_label=prediction["candidate_label"],
                               similarity=prediction["similarity"])
                writer.writerow(row)
            for predicted_index, prediction in enumerate(predictions):
                if predicted_index not in matched_predictions:
                    writer.writerow({"image": sample["identifier"], "split": sample["split"], "status": "false_positive",
                                     "predicted_label": prediction["label"], "candidate_label": prediction["candidate_label"],
                                     "similarity": prediction["similarity"]})
            record = {"image": sample["identifier"], "split": sample["split"], "image_sha256": identity["image_sha256"],
                      "width": image.width, "height": image.height, "ground_truth": truth,
                      "detections": predictions, "elapsed_seconds": time.perf_counter() - before,
                      "matches": [{"truth_index": actual, "prediction_index": predicted, "iou": overlap}
                                  for actual, predicted, overlap in matches]}
            reporter.record(record, image)
            crop_reporter.record({**record, "detections": crop_predictions, "matches": []})
            if number == 1 or number % 25 == 0 or number == len(samples):
                print(f"Evaluated {number}/{len(samples)} images; {counts['ground_truth']} annotated vehicles; "
                      f"{counts['matched']} detected; {time.perf_counter() - started:.1f}s", file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    reporter.finish(elapsed)
    crop_reporter.finish(elapsed)
    result = {"output_directory": str(directory), "images": counts["images"], "failed_images": 0,
              "images_per_split": dict(split_counts), "ground_truth_objects": counts["ground_truth"],
              "positive_images": counts["positive_images"], "negative_images": counts["negative_images"],
              "predictions": counts["predictions"], "unique_image_hashes": len(digests),
              "exact_duplicate_images": sum(count - 1 for count in digests.values()),
              "iou_threshold": iou_threshold, "similarity_threshold": classifier.db.threshold,
              "detection": precision_recall(counts["matched"], counts["predictions"], counts["ground_truth"]),
              "similarity_on_ground_truth_crops": crop_scores.result(),
              "similarity_on_matched_detections": matched_scores.result(),
              "end_to_end": precision_recall(counts["end_to_end_correct"], counts["predictions"], counts["ground_truth"]),
              "end_to_end_per_class": {label: {"objects": values["objects"], "detected": values["detected"],
                                               "correct": values["correct"],
                                               "recall": ratio(values["correct"], values["objects"])}
                                       for label, values in class_counts.items()},
              "elapsed_seconds": elapsed}
    write_json(directory / "metrics.json", result)
    return result
