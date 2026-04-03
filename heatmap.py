"""Heatmap visualization manager for 4D Gaussian attributes."""

import numpy as np
from enum import IntEnum
from loader_4d import Checkpoint4DData


class HeatmapMode(IntEnum):
    NONE = 0           # Normal rendering
    VELOCITY = 1       # Velocity magnitude
    TEMPORAL_OPA = 2   # Temporal opacity at current time
    DURATION = 3       # Duration (exp of log-duration)
    BASE_OPACITY = 4   # Base opacity (sigmoid)
    GRADIENT = 5       # Gradient accumulator (training bridge)


def generate_turbo_lut() -> np.ndarray:
    """Generate turbo colormap LUT as [256, 3] float32 array."""
    # Turbo colormap keypoints (subset, interpolated)
    turbo_data = [
        [0.18995, 0.07176, 0.23217],
        [0.22500, 0.16354, 0.45096],
        [0.25107, 0.25237, 0.63374],
        [0.26816, 0.33929, 0.77642],
        [0.27628, 0.42118, 0.88563],
        [0.25862, 0.50543, 0.93906],
        [0.21382, 0.58549, 0.92170],
        [0.15844, 0.65867, 0.82789],
        [0.12129, 0.72312, 0.66510],
        [0.13098, 0.77680, 0.46770],
        [0.22614, 0.81910, 0.27106],
        [0.39346, 0.84890, 0.12394],
        [0.57120, 0.86500, 0.05950],
        [0.72410, 0.86100, 0.07920],
        [0.84299, 0.83700, 0.13600],
        [0.92786, 0.79400, 0.21300],
        [0.97610, 0.73200, 0.30100],
        [0.99324, 0.65200, 0.39200],
        [0.98320, 0.55700, 0.48100],
        [0.94960, 0.45100, 0.56200],
        [0.89240, 0.34200, 0.63200],
        [0.81710, 0.24300, 0.68800],
        [0.72830, 0.15800, 0.72500],
        [0.63070, 0.09400, 0.73700],
        [0.52960, 0.05400, 0.72000],
        [0.43100, 0.03300, 0.67600],
        [0.34000, 0.02200, 0.60800],
        [0.26000, 0.01600, 0.52000],
        [0.19000, 0.01500, 0.42000],
        [0.13000, 0.01400, 0.32000],
        [0.09000, 0.01300, 0.23000],
        [0.05000, 0.01200, 0.15000],
    ]
    keypoints = np.array(turbo_data, dtype=np.float32)
    n_keys = len(keypoints)
    lut = np.zeros((256, 3), dtype=np.float32)
    for i in range(256):
        t = i / 255.0 * (n_keys - 1)
        idx = int(t)
        frac = t - idx
        if idx >= n_keys - 1:
            lut[i] = keypoints[-1]
        else:
            lut[i] = keypoints[idx] * (1 - frac) + keypoints[idx + 1] * frac
    return lut


class HeatmapManager:
    """Manages heatmap mode and scalar field computation."""

    def __init__(self):
        self.mode: HeatmapMode = HeatmapMode.NONE
        self.turbo_lut: np.ndarray = generate_turbo_lut()

    def compute_scalar_field(self, data: Checkpoint4DData,
                             current_time: float) -> np.ndarray | None:
        """Compute per-gaussian scalar values for current mode. Returns [N]."""
        if self.mode == HeatmapMode.NONE:
            return None

        if self.mode == HeatmapMode.VELOCITY:
            return np.linalg.norm(data.velocities, axis=-1)

        if self.mode == HeatmapMode.TEMPORAL_OPA:
            dt = current_time - data.times[:, 0]
            s = np.maximum(np.exp(data.durations[:, 0]), 0.02)
            return np.exp(-0.5 * (dt / s) ** 2)

        if self.mode == HeatmapMode.DURATION:
            return np.exp(data.durations[:, 0])

        if self.mode == HeatmapMode.BASE_OPACITY:
            # sigmoid of logit
            return 1.0 / (1.0 + np.exp(-data.opacities[:, 0]))

        if self.mode == HeatmapMode.GRADIENT:
            if hasattr(data, 'grad_accum') and data.grad_accum is not None:
                return data.grad_accum.copy()
            return np.zeros(data.n_gaussians, dtype=np.float32)

        return None

    def normalize_and_map(self, scalars: np.ndarray) -> np.ndarray:
        """Normalize scalars to [0,1] and map to RGB via turbo LUT.
        Returns [N, 3] float32."""
        smin = scalars.min()
        smax = scalars.max()
        rng = smax - smin
        if rng < 1e-8:
            norm = np.full_like(scalars, 0.5)
        else:
            norm = (scalars - smin) / rng
        norm = np.clip(norm, 0.0, 1.0)

        indices = (norm * 255).astype(np.int32)
        indices = np.clip(indices, 0, 255)
        return self.turbo_lut[indices]  # [N, 3]

    def get_range(self, scalars: np.ndarray) -> tuple[float, float]:
        """Return (min, max) for shader-side normalization.
        Uses percentile clipping to handle extreme outliers."""
        if self.mode == HeatmapMode.GRADIENT:
            # Gradient values are extremely skewed — use p5/p95
            vmin = float(np.percentile(scalars, 5))
            vmax = float(np.percentile(scalars, 95))
        else:
            vmin = float(np.percentile(scalars, 1))
            vmax = float(np.percentile(scalars, 99))
        if vmax - vmin < 1e-8:
            vmax = vmin + 1.0
        return vmin, vmax
