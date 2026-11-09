"""Planned dataset and pretrained-weight preparation.

Read the Roboflow workspace/project/version and download a fixed YOLO-format
export to persistent storage. Accept existing local exports. Preserve the
export's actual class order and validate canonical recognition label names.

Cache the VisDrone YOLO checkpoint separately from the optional dataset.
Only download VisDrone when explicitly requested with --include-visdrone.
Use the official conversion rules and configure the destination explicitly.
Record source revisions and artifact hashes for reproducible preparation.
No detector-training operation belongs to this initial implementation.
"""
