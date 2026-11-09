# Exploratory analysis

This directory is reserved for reproducible dataset exploration and presentation figures.

Planned VisDrone analysis includes class frequencies, bounding-box area relative to image area, image resolution, representative scenes, and qualitative detector results. Downloading the dataset will require the explicit `--include-visdrone` preparation option.

`prepare --include-visdrone --visdrone-splits val` prepares a smaller first sample. Converted labels, preserved original annotations, class counts, and source archive hashes are stored under `data/visdrone/` for reproducible exploration.

Read persistent assets from `data/` and write generated figures to `outputs/`. Distinguish simulation-generated reference crops from real forest-flight validation images in figures and captions. Keep final forest evaluation data independent of gallery selection and threshold calibration so that transfer from simulation to real imagery can be assessed.
