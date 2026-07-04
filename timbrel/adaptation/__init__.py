"""Speaker adaptation: few-shot enrolment and parameter-efficient fine-tuning."""

from timbrel.adaptation.enroll import SpeakerEnroller, enroll
from timbrel.adaptation.finetune import (
    adapt,
    conditional_ln_parameters,
    freeze_for_adaptation,
)

__all__ = [
    "SpeakerEnroller",
    "enroll",
    "adapt",
    "conditional_ln_parameters",
    "freeze_for_adaptation",
]

