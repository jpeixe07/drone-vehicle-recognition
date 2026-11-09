"""Planned optional augmentation of labeled reference crops.

Preserve labels while generating seeded rotations, flips, brightness and
color changes, and mild geometric/blur variations. Record parent images
and transformation settings. Never augment the final evaluation dataset.
Generating variants is data preparation, not training EfficientNet-B3.
"""
