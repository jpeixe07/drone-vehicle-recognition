"""Planned offline orchestration for image and directory inputs.

Load configuration, models, and the compatible reference index once per run.
Invoke the central recognizer and send predictions to reporting. Optional
rotated-image inference must invert transforms, clip boxes to the original
image, and merge duplicates before scoring/reporting individual vehicles.
Aggregate multiple observations only when they are explicitly associated
with the same vehicle; unrelated image detections must not share votes.
"""
