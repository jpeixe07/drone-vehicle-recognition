"""Explicit preparation of pretrained weights and optional public/private data."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image
import yaml

from .configuration import ProjectSettings, read_yaml
from .reporting import write_json


def prepare_weights(settings: ProjectSettings, force: bool = False) -> dict:
    from huggingface_hub import HfApi, hf_hub_download
    from .similarity_efficientnet import backbone_spec, checkpoint_path, checkpoint_sha256, file_sha256

    detector = settings.config["detector"]
    target = settings.path(detector["weights"])
    manifest_path = target.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"schema_version": 1}
    sources = manifest.setdefault("detectors", {})
    legacy = manifest.pop("detector", None)
    if legacy and legacy.get("file"):
        sources.setdefault(legacy["file"], legacy)
    previous = sources.get(target.name, {})
    if target.is_file() and not force:
        if previous.get("source") == "huggingface":
            expected_source = (detector["huggingface_repo"], detector["huggingface_filename"],
                               detector.get("huggingface_revision", "main"))
            actual_source = (previous.get("repo_id"), previous.get("filename"), previous.get("requested_revision"))
            if actual_source != expected_source:
                raise ValueError("Cached detector source differs from configuration; choose a different weights path or use --force")
        digest = file_sha256(target)
        if previous and previous.get("sha256") != digest:
            raise ValueError("Detector checksum differs from its preparation manifest; use --force to download it again")
        source = previous or {"source": "local", "file": target.name, "sha256": digest}
    else:
        repo = detector["huggingface_repo"]
        revision = detector.get("huggingface_revision", "main")
        resolved = HfApi().model_info(repo, revision=revision).sha
        cached = hf_hub_download(repo_id=repo, filename=detector["huggingface_filename"], revision=resolved,
                                 force_download=force)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        shutil.copyfile(cached, temporary)
        temporary.replace(target)
        source = {"source": "huggingface", "repo_id": repo, "revision": resolved,
                  "requested_revision": revision, "filename": detector["huggingface_filename"],
                  "file": target.name, "sha256": file_sha256(target)}
    recognition = settings.config["recognition"]
    _, checkpoint, dimension = backbone_spec(recognition["backbone"], recognition["weights"])
    path = checkpoint_path(checkpoint)
    if not path.is_file():
        checkpoint.get_state_dict(progress=True, check_hash=True)
    extractor = {"backbone": recognition["backbone"], "weights": checkpoint.name, "url": checkpoint.url,
                 "file": str(path), "sha256": checkpoint_sha256(checkpoint), "feature_dimension": dimension}
    sources[target.name] = source
    manifest.setdefault("extractors", {})[recognition["backbone"]] = extractor
    write_json(manifest_path, manifest)
    return {"detector": source, "extractor": extractor}


def download_file(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    try:
        request = Request(url, headers={"User-Agent": "aerial-vehicle-recognition/0.1"})
        with urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
            expected = response.headers.get("Content-Length")
            count = 0
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
                count += len(chunk)
            if expected is not None and count != int(expected):
                raise OSError(f"Incomplete download: expected {expected} bytes, received {count}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def extract_zip(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f"Archive contains a path outside its extraction directory: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Archive contains a symbolic link")
        zipped.extractall(root)


def convert_visdrone_split(source: Path, destination: Path, split: str, names: list[str]) -> dict:
    """Convert valid original boxes, retaining raw annotations for ignored regions."""
    images = destination / "images" / split
    labels = destination / "labels" / split
    annotations = destination / "annotations" / split
    for directory in [images, labels, annotations]:
        directory.mkdir(parents=True, exist_ok=True)
    counts = dict.fromkeys(names, 0)
    paths = sorted(path for path in (source / "images").iterdir() if path.suffix.lower() == ".jpg")
    if not paths:
        raise ValueError(f"No VisDrone images found: {source}")
    for image_path in paths:
        annotation = source / "annotations" / image_path.with_suffix(".txt").name
        with Image.open(image_path) as image:
            width, height = image.size
        converted = []
        for number, row in enumerate(annotation.read_text().splitlines(), 1):
            if not row.strip():
                continue
            fields = row.strip().rstrip(",").split(",")
            if len(fields) < 8:
                raise ValueError(f"Invalid annotation row {number}: {annotation}")
            x, y, box_width, box_height = map(float, fields[:4])
            score, class_id = float(fields[4]), int(fields[5])
            if not all(math.isfinite(value) for value in [x, y, box_width, box_height, score]):
                raise ValueError(f"Non-finite annotation row {number}: {annotation}")
            if score == 0 or class_id < 1 or class_id > len(names):
                continue
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(float(height), y + box_height)
            if x2 <= x1 or y2 <= y1:
                continue
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
            converted.append(f"{class_id - 1} " + " ".join(f"{value:.8f}" for value in values) + "\n")
            counts[names[class_id - 1]] += 1
        (labels / annotation.name).write_text("".join(converted), encoding="utf-8")
        shutil.copyfile(annotation, annotations / annotation.name)
        shutil.move(str(image_path), str(images / image_path.name))
    return {"images": len(paths), "class_counts": counts}


def prepare_visdrone(settings: ProjectSettings, splits: list[str] | None = None, force: bool = False) -> dict:
    from ultralytics.utils import ASSETS_URL
    from .similarity_efficientnet import file_sha256

    config = settings.datasets["visdrone"]
    destination = settings.path(config["directory"])
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"schema_version": 1, "splits": {}}
    sources = {"train": "VisDrone2019-DET-train", "val": "VisDrone2019-DET-val", "test": "VisDrone2019-DET-test-dev"}
    for split in splits or ["train", "val", "test"]:
        if split not in sources:
            raise ValueError(f"Unsupported VisDrone split: {split}")
        if split in manifest["splits"] and not force:
            if (destination / "images" / split).is_dir() and (destination / "labels" / split).is_dir():
                continue
        if not force and (destination / "images" / split).exists():
            raise ValueError(f"Existing {split} split has no complete manifest; use --force or a fresh dataset directory")
        folder = sources[split]
        url = f"{ASSETS_URL}/{folder}.zip"
        archive = settings.root / "artifacts/cache/datasets" / f"{folder}.zip"
        if not archive.is_file():
            download_file(url, archive)
        with tempfile.TemporaryDirectory(dir=destination, prefix=".extract-") as temporary:
            extract_zip(archive, Path(temporary))
            info = convert_visdrone_split(Path(temporary) / folder, destination, split, config["labels"])
        manifest["splits"][split] = {**info, "url": url, "archive_sha256": file_sha256(archive)}
        write_json(manifest_path, manifest)
    # Omitting path lets Ultralytics resolve image paths relative to this YAML.
    dataset = {split: f"images/{split}" for split in sources if split in manifest["splits"]}
    dataset["names"] = dict(enumerate(config["labels"]))
    (destination / "data.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    return {"directory": str(destination), **manifest}


def dataset_names(config: dict) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        indexed = {int(key): value for key, value in names.items()}
        if sorted(indexed) != list(range(len(indexed))):
            raise ValueError("Dataset class IDs must be contiguous and start at zero")
        names = [indexed[key] for key in range(len(indexed))]
    if not isinstance(names, list) or not names or not all(isinstance(value, str) for value in names):
        raise ValueError("Dataset YAML must define a nonempty names list or indexed mapping")
    if len(names) != len(set(names)):
        raise ValueError("Dataset class names must be unique")
    if "nc" in config and int(config["nc"]) != len(names):
        raise ValueError("Dataset nc does not match the number of class names")
    return names


def prepare_forest(settings: ProjectSettings, force: bool = False) -> dict:
    config = settings.datasets["forest_vehicles"]
    destination = settings.path(config["directory"])
    workspace = os.environ.get(config["workspace_env"]) or config.get("workspace")
    revision = os.environ.get(config["version_env"]) or config.get("version")
    project = os.environ.get("ROBOFLOW_PROJECT") or config["project"]
    if revision is not None:
        if isinstance(revision, bool) or not str(revision).isdigit() or int(revision) < 1:
            raise ValueError("Choose a positive, fixed Roboflow dataset version")
        revision = int(revision)
    source = {"workspace": workspace, "project": project, "version": revision,
              "export_format": config["export_format"]}
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None and not force and any(manifest.get(key) != value for key, value in source.items()):
        raise ValueError("Cached forest dataset source differs from configuration; use --force or a fresh dataset directory")
    if not (destination / "data.yaml").is_file() or force:
        api_key = os.environ.get(config["api_key_env"])
        if not workspace or not revision or not api_key:
            raise ValueError("Configure a Roboflow workspace and version and set "
                             f"{config['api_key_env']} in .env or the environment, or provide a local export")
        if destination.exists() and any(path.name != ".gitkeep" for path in destination.iterdir()) and not force:
            raise ValueError("Forest data directory is nonempty but lacks data.yaml; use a fresh directory or --force")
        try:
            from roboflow import Roboflow
        except ImportError:
            raise RuntimeError("Install project dependencies with python -m pip install -e .") from None

        # Download separately so failed refreshes preserve the previous export and
        # successful refreshes cannot retain files from an older version.
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".forest-download-") as temporary:
            export = Path(temporary) / "export"
            Roboflow(api_key=api_key).workspace(workspace).project(project).version(revision).download(
                model_format=config["export_format"], location=str(export), overwrite=True,
            )
            _forest_names(settings, export / "data.yaml")
            _portable_forest_yaml(export / "data.yaml")
            write_json(export / "manifest.json", source)
            if (destination / ".gitkeep").is_file():
                shutil.copyfile(destination / ".gitkeep", export / ".gitkeep")
            previous = Path(temporary) / "previous"
            if destination.exists():
                destination.rename(previous)
            try:
                export.rename(destination)
            except OSError:
                if previous.exists():
                    previous.rename(destination)
                raise
    names = _forest_names(settings, destination / "data.yaml")
    # Migrate cached exports too, without another download or credentials.
    _portable_forest_yaml(destination / "data.yaml")
    return {"directory": str(destination), "data_yaml": str(destination / "data.yaml"),
            "names": names, "role": config["role"], "source": source if manifest_path.is_file() else None}


def _portable_forest_yaml(path: Path) -> None:
    dataset = read_yaml(path)
    original = dict(dataset)
    previous_root = Path(dataset.get("path") or path.parent)
    for split in ("train", "val", "test"):
        if not dataset.get(split):
            continue
        values = dataset[split] if isinstance(dataset[split], list) else [dataset[split]]
        portable = []
        for value in values:
            source = Path(value)
            if source.is_absolute():
                for root in (path.parent, previous_root):
                    if root.is_absolute() and source.is_relative_to(root):
                        source = source.relative_to(root)
                        break
            if source.parts[:1] == ("..",):
                local = Path(*source.parts[1:])
                if (path.parent / local).exists():
                    source = local
            portable.append(str(source))
        dataset[split] = portable if isinstance(dataset[split], list) else portable[0]
    # Ultralytics resolves missing `path` relative to the YAML itself. `path: .`
    # would instead resolve against its global datasets directory.
    dataset.pop("path", None)
    if dataset != original:
        path.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")


def _forest_names(settings: ProjectSettings, path: Path) -> list[str]:
    names = dataset_names(read_yaml(path))
    invalid = set(names) - set(settings.labels) - {settings.config["recognition"]["unknown_label"]}
    if invalid:
        raise ValueError(f"Roboflow labels do not match the recognition catalog: {', '.join(sorted(invalid))}")
    return names


def prepare_assets(settings: ProjectSettings, include_visdrone: bool = False, include_forest: bool = False,
                   splits: list[str] | None = None, force: bool = False) -> dict:
    result = {"weights": prepare_weights(settings, force)}
    if include_visdrone:
        result["visdrone"] = prepare_visdrone(settings, splits, force)
    if include_forest:
        result["forest_vehicles"] = prepare_forest(settings, force)
    return result
