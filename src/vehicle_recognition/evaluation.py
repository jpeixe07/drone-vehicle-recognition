"""Planned evaluation against the final annotated forest dataset.

Separate vehicle localization metrics, fine-grained recognition on matched
vehicles, and end-to-end results. Preserve unknown predictions and measure
their behavior only when suitable out-of-catalog ground truth is present.
Threshold calibration and reference selection must not use final test data.
Detector-only broad labels and specific vehicle-model labels are different
tasks and must not be presented as directly comparable class accuracies.
"""
