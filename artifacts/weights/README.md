# Versioned detector checkpoints

These two checkpoints are included in Git and copied into the Docker image.
They remain frozen during evaluation.

| File | Source | SHA-256 |
| --- | --- | --- |
| `visdrone-yolov8s.pt` | Hugging Face `dronefreak/visdrone-yolov8s`, file `best.pt`, revision `cbcca22c6388563fee903e67794dd4ee7755f4a1` | `29dae68ac5028cedac1f396333ac89cd9c74232691155243d70d9a1c16cdb24e` |
| `vehicle_detection_yolo26s_MVRSD_VIS.pt` | User-provided YOLO26s MVRSD/VIS checkpoint, serialized with Ultralytics 8.4.115; one `vehicle` class | `489e0d8e90a81575ec726bef972f6327c7b404ad8aa53f7139cba4762d70d362` |

`configs/default.yaml` selects YOLOv8; `configs/yolo26s_mvrsd_vis.yaml` selects
YOLO26. Preparation verifies existing checksums using the generated local
`manifest.json`, which is ignored because it contains machine-specific cache
paths. The SHA-256 values above identify the committed originals.

The official EfficientNet-B3 ImageNet-1K checkpoint is downloaded by `prepare`
into the ignored Torch cache. It is not duplicated here. Dataset exports,
reference indexes, caches, and evaluation outputs remain outside Git.
