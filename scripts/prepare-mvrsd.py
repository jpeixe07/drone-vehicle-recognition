#!/usr/bin/env python3
"""Download/cache the official MVRSD RAR and extract its original labels.

Usage: python scripts/prepare-mvrsd.py --download
Requires the `unrar` executable; alternatively pass --archive to a local RAR.
No annotation conversion, augmentation, split creation, or training is performed.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen

URL = "https://china.scidb.cn/download?fileId=afddb6975d4fc876fdd7c19d490d42b9"
DOI = "https://doi.org/10.57760/sciencedb.j00240.00036"
ARCHIVE_BYTES = 987739200
ARCHIVE_SHA256 = "dd5e3216b944831bf1710a4d6a28b866d83d37d54322e8fc221faf42ca007343"


def download(target: Path, workers: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == ARCHIVE_BYTES:
        return
    parts = target.parent / "mvrsd-rar-parts"
    parts.mkdir(exist_ok=True)
    chunk_size = 16 * 1024 * 1024
    count = math.ceil(ARCHIVE_BYTES / chunk_size)

    def fetch(number):
        start = number * chunk_size
        end = min(ARCHIVE_BYTES, start + chunk_size) - 1
        path = parts / f"{number:04d}"
        if path.is_file() and path.stat().st_size == end - start + 1:
            return
        for attempt in range(3):
            try:
                request = Request(URL, headers={"Range": f"bytes={start}-{end}"})
                with urlopen(request, timeout=90) as response:
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
                        raise OSError("Official server did not honor the requested byte range")
                    temporary = path.with_suffix(".part")
                    with temporary.open("wb") as handle:
                        shutil.copyfileobj(response, handle, 1024*1024)
                if temporary.stat().st_size != end-start+1:
                    raise OSError("Incomplete archive chunk")
                temporary.replace(path)
                return
            except OSError:
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for number, future in enumerate(as_completed([executor.submit(fetch, i) for i in range(count)]), 1):
            future.result()
            print(f"Archive chunks: {number}/{count}", flush=True)
    assembled = target.with_suffix(".assembled")
    with assembled.open("wb") as output:
        for number in range(count):
            with (parts / f"{number:04d}").open("rb") as handle:
                shutil.copyfileobj(handle, output, 1024*1024)
    if assembled.stat().st_size != ARCHIVE_BYTES:
        raise OSError("Incomplete official archive")
    assembled.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    root = args.root.resolve()
    archive = args.archive or root / "artifacts/cache/datasets/MVRSD_dataset.rar"
    if not archive.is_file() and not args.download:
        parser.error("Use --download or --archive /path/to/MVRSD_dataset.rar")
    if args.download and not args.archive:
        download(archive, args.workers)
    unrar = shutil.which("unrar")
    if not unrar:
        raise SystemExit("Install the system 'unrar' executable, or extract the archive manually into data/mvrsd/")
    digest = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    if not args.archive and digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Official archive checksum changed; inspect its version before extracting")
    target = root / "data/mvrsd"
    manifest = target / "source.json"
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if previous.get("archive_sha256") == digest.hexdigest() and (target / "MVRSD_dataset/data/images").is_dir():
            print("Reusing extracted official dataset:", target)
            return
    members = subprocess.check_output([unrar, "lb", "-p-", str(archive)], text=True).splitlines()
    for member in members:
        value = member.replace("\\", "/")
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive or ".." in PurePosixPath(value).parts:
            raise ValueError(f"Archive contains an unsafe path: {member}")
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target, prefix=".extract-") as temporary:
        subprocess.run([unrar, "x", "-p-", "-o-", "-idq", str(archive.resolve()), temporary+"/"], check=True)
        for entry in Path(temporary).iterdir():
            destination = target / entry.name
            if destination.exists():
                raise FileExistsError(f"Existing dataset directory has no matching manifest: {destination}")
            entry.rename(destination)
    result = {"doi": DOI, "download_url": URL, "archive_bytes": archive.stat().st_size,
              "archive_sha256": digest.hexdigest(), "annotation_conversion": False,
              "extraction": "original archive layout, XML and TXT preserved"}
    manifest.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
