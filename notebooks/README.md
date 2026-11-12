# Exploratory analysis

Keep dataset-specific exploration separate from the comparison:

| Notebook | Purpose | Generated artifacts |
| --- | --- | --- |
| `01_eda_visdrone.ipynb` | Existing VisDrone train/val analysis and simulated reference gallery | `outputs/eda_visdrone/` |
| `02_eda_mvrsd.ipynb` | MVRSD train/val class frequencies, box geometry/COCO scales, image resolution, scene density, spatial distribution, label quality, example scenes, qualitative detections, and reference gallery | `outputs/eda_mvrsd/` |
| `03_comparacao_visdrone_mvrsd.ipynb` | Compare compatible measurements while preserving each dataset's taxonomy and scope | `outputs/eda_comparison/` |

## Environment

Use the installed project environment as the notebook kernel:

```bash
python -m pip install -e . -r requirements-notebooks.txt
```

The notebooks locate the repository from either its root or `notebooks/`.
For a standalone MVRSD run, the same implementation is available as:

```bash
python -m vehicle_recognition.eda_mvrsd --device cuda
# Without qualitative detector inference:
python -m vehicle_recognition.eda_mvrsd --skip-detector
```

## MVRSD acquisition

The official source is [ScienceDB, DOI 10.57760/sciencedb.j00240.00036](https://doi.org/10.57760/sciencedb.j00240.00036),
with [CSTR file identifier](https://cstr.cn/31253.11.sciencedb.j00240.00036.011B082C).
The original [publication](https://www.cjig.cn/en/article/doi/10.11834/jig.230884/) and
[authors' repository](https://github.com/baidongls/MVRSD) describe the five classes
SMV, LMV, AFV, MCV, and CV. Published counts and nominal 0.3 m/pixel resolution
are references, not substitutes for measurements of the downloaded archive.

```bash
# Requires the system `unrar` executable. Public download; no API key required.
python scripts/prepare-mvrsd.py --download
# Or use an existing archive:
python scripts/prepare-mvrsd.py --archive /path/to/MVRSD_dataset.rar
```

The download is approximately 988 MB. HTTP byte ranges are cached for resume;
the assembled RAR stays in `artifacts/cache/datasets/`. Extraction preserves
the original layout under `data/mvrsd/MVRSD_dataset/`, including
`data/images/{train,val}`, `data/labels/{train,val}/*.txt`, and the matching
`data/labels/{train,val}/xml/*.xml` files.
`data/mvrsd/source.json` records the archive hash and source. Alternatively,
extract manually into `data/mvrsd/` or change `DATA_ROOT` in notebook 02.
Raw data, archive chunks, and generated CSV/JSON/PNG artifacts are ignored by Git.

XML and TXT encode the same objects; the loader prefers named XML and never
adds both annotation copies. Numeric-only YOLO exports require `data.yaml`
or the notebook's explicit `YOLO_CLASS_NAMES` mapping. Publication class order
is not assumed to be YOLO ID order. Orphan labels, missing annotations,
invalid boxes, duplicates, and excluded unannotated panoramas are audited.
Missing labels are not interpreted as negative images, and absent occlusion
fields are not interpreted as zero occlusion.

## VisDrone inputs and comparison provenance

`prepare --include-visdrone --visdrone-splits val` prepares a smaller first sample.
The existing full EDA expects train and val, preserves original annotation
metadata, and exports class/geometry/density summaries:

```bash
python -m vehicle_recognition prepare --include-visdrone --visdrone-splits train val
python -m vehicle_recognition.eda_visdrone
```

Run notebook 02 before notebook 03. The comparison prefers native VisDrone
JSON/CSV artifacts. If they are unavailable, it recovers only saved executed
HTML tables and JSON stdout from notebook 01, records its SHA-256, and labels
the source as a historical notebook snapshot. It does not extract measurements
from Markdown prose or imply that raw VisDrone data was reprocessed. Saved
rounding is preserved; missing raw distributions are not reconstructed.

The comparison keeps ten VisDrone classes separate from the five MVRSD classes.
Vehicle-only geometry uses VisDrone car/van/truck/bus and all five MVRSD classes;
categories are not mapped one-to-one. It does not average class medians or
present qualitative counts on possible training data as validation accuracy.

Read persistent assets from `data/` and write generated figures to `outputs/`.
Keep simulated reference crops and final forest evaluation images distinct.
These notebooks do not train the models, choose new references, or calibrate
thresholds on the final forest evaluation dataset.
