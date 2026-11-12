# Implementation backlog

## Initial offline pipeline

- [x] Implement the EfficientNet-B3 feature extractor in `similarity_efficientnet.py` using the official ImageNet-1K checkpoint and normalized pooled features.
- [x] Implement individual-reference cosine matching, maximum similarity per class, top candidates, and unknown rejection.
- [x] Implement YOLO integration with class-name-based filtering and configurable crop margins.
- [x] Build/load gallery indexes with checkpoint, preprocessing, and gallery compatibility metadata.
- [x] Add pretrained-weight preparation with recorded source revisions and hashes.
- [x] Add the optional Roboflow downloader and accept an existing local export.
- [x] Add optional VisDrone downloading/conversion through `--include-visdrone`.
- [x] Implement seeded reference augmentation with parent-image metadata.
- [x] Implement offline directory inference and JSON/CSV/image reports.
- [x] Implement optional rotation inference with coordinate inversion and duplicate merging.
- [x] Evaluate detection, model recognition, and end-to-end results separately; report unknown rejection as unavailable without annotated unknowns.
- [x] Exercise the final Roboflow export with frozen weights/references and save per-object results.
- [ ] Pin a complete dependency resolution.

## Dataset exploration

- [x] Add VisDrone class/box-size distributions and example scenes.
- [x] Add a separate MVRSD EDA notebook using the original ScienceDB archive and named XML annotations.
- [x] Add a VisDrone/MVRSD comparison notebook with explicit data provenance and distinct taxonomies.
- [x] Generate presentation figures from saved, reproducible analysis runs.

## Future detector experiments

- [ ] Prepare DroneVehicle RGB data, deciding explicitly between oriented boxes and rectangular-box conversion.
- [ ] Handle image borders and annotation coordinates consistently during conversion.
- [ ] Train a detector on DroneVehicle; training is outside the initial project scope.
- [ ] Compare VisDrone- and DroneVehicle-trained detectors with the same EfficientNet-B3 gallery and evaluation protocol.
- [ ] Consider extractor adaptation only after studying errors from the pretrained-feature baseline.
