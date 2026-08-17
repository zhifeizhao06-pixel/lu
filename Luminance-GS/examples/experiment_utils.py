"""Pure helpers shared by supplementary experiment code and tests."""

import math
from typing import Literal, Optional

Schedule = Literal["fixed", "step", "linear", "smoothstep", "exponential"]


def curriculum_weight(
    step: int,
    start: int,
    end: int,
    schedule: Schedule = "smoothstep",
    exponential_rate: float = 5.0,
    step_iteration: Optional[int] = None,
) -> float:
    """Return a bounded confidence-control weight for a training step."""
    if end <= start:
        raise ValueError("end must be greater than start")
    if exponential_rate <= 0:
        raise ValueError("exponential_rate must be positive")
    if schedule == "fixed":
        return 1.0
    if schedule == "step":
        transition = (start + end) // 2 if step_iteration is None else step_iteration
        return 0.0 if step < transition else 1.0

    progress = min(max((step - start) / (end - start), 0.0), 1.0)
    if schedule == "linear":
        return progress
    if schedule == "smoothstep":
        return progress * progress * (3.0 - 2.0 * progress)
    if schedule == "exponential":
        numerator = math.expm1(exponential_rate * progress)
        denominator = math.expm1(exponential_rate)
        return numerator / denominator
    raise ValueError(f"Unknown schedule: {schedule}")
