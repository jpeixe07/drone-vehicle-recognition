"""Planned per-run JSON/CSV predictions, annotated images, and summaries.

Each vehicle record contains source image, original-image bounding box,
detector class/confidence, recognized model or unknown, similarity, and
ranked candidates. Store timing, the resolved configuration, dependency
versions, detector/extractor identifiers, and gallery fingerprint. Keep
credentials out of saved configuration and retain rejected predictions.
"""
