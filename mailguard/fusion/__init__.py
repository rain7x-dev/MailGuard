"""Fusion package: combine seven signal scores into one explainable verdict.

`ebm_fusion` holds the additive model and the calibration slot, `verdict`
holds the probability to class and action mapping. Re-exported here so
callers can write `from mailguard.fusion import fuse`.
"""
from __future__ import annotations

from mailguard.fusion.ebm_fusion import (
    FEATURE_NAMES,
    INTERACTION_WEIGHTS,
    SIGNAL_ORDER,
    SIGNAL_WEIGHTS,
    build_feature_vector,
    calibrate,
    dominant_signals,
    fuse,
    interaction_contributions,
    logit_of,
    model_status,
    signal_contributions,
)
from mailguard.fusion.verdict import (
    ACTIONS,
    THRESHOLDS,
    VERDICT_CLASSES,
    band_summary,
    describe_action,
    to_action,
    to_verdict_class,
)

__all__ = [
    "ACTIONS",
    "FEATURE_NAMES",
    "INTERACTION_WEIGHTS",
    "SIGNAL_ORDER",
    "SIGNAL_WEIGHTS",
    "THRESHOLDS",
    "VERDICT_CLASSES",
    "band_summary",
    "build_feature_vector",
    "calibrate",
    "describe_action",
    "dominant_signals",
    "fuse",
    "interaction_contributions",
    "logit_of",
    "model_status",
    "signal_contributions",
    "to_action",
    "to_verdict_class",
]
