#!/usr/bin/env python3
"""Assemble the shared reference gallery without overwriting different inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/recognition_comparison_margin5/gallery_82"))
    parser.add_argument("--classes-file", type=Path, default=Path("configs/classes_comparison.yaml"))
    args = parser.parse_args()
    root = args.root.resolve()
    target = (root / args.output).resolve()
    labels = yaml.safe_load((root / args.classes_file).read_text())["labels"]
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    entries = []
    for directory, prefix in [("reference_dataset", ""), ("reference_dataset_additional", "additional__")]:
        source = root / "data" / directory
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            relative = path.relative_to(source)
            if len(relative.parts) != 2:
                raise ValueError(f"Expected class/image layout: {path}")
            if relative.parts[0] not in labels:
                continue
            destination = Path(relative.parts[0]) / (prefix + path.name)
            entries.append((path, destination, hashlib.sha256(path.read_bytes()).hexdigest()))
    if not entries or len({str(item[1]) for item in entries}) != len(entries):
        raise ValueError("Empty gallery or conflicting image names")
    expected = {str(relative): digest for _, relative, digest in entries}
    if target.exists():
        actual = {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in target.rglob("*") if p.is_file() and p.suffix.lower() in suffixes}
        if actual != expected:
            raise ValueError("Existing gallery differs from the source images; choose a new --output")
    else:
        for path, relative, _ in entries:
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, output)
    manifest = {"images": len(entries), "files": [
        {"source": str(path.relative_to(root)), "gallery_path": str(relative), "sha256": digest}
        for path, relative, digest in entries]}
    (target.parent / "gallery_sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified shared gallery: {len(entries)} images in {target}")


if __name__ == "__main__":
    main()
