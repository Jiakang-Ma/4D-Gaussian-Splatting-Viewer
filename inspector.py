"""Gaussian attribute inspector — ray-cast selection and property display."""

import numpy as np
from typing import Optional
from loader_4d import Checkpoint4DData


class GaussianInspector:
    """Select and inspect individual Gaussian properties."""

    def __init__(self):
        self.selected_index: Optional[int] = None
        self.highlight_color = (1.0, 0.3, 0.0)

    def pick(self, ray_origin: np.ndarray, ray_dir: np.ndarray,
             positions: np.ndarray, scales: np.ndarray,
             opacities: np.ndarray, opacity_threshold: float = 0.01
             ) -> Optional[int]:
        """Find the closest visible Gaussian to the ray.

        Args:
            ray_origin: [3] camera position
            ray_dir: [3] normalized ray direction
            positions: [N, 3] current positions
            scales: [N, 3] Gaussian scales
            opacities: [N] final opacities (after temporal)
            opacity_threshold: minimum opacity to consider visible

        Returns:
            Index of closest Gaussian, or None if nothing hit.
        """
        # Filter visible
        visible_mask = opacities > opacity_threshold
        if not np.any(visible_mask):
            self.selected_index = None
            return None

        visible_idx = np.where(visible_mask)[0]
        pos = positions[visible_idx]  # [M, 3]

        # Point-to-ray distance: d = ||(p - o) - ((p - o) . dir) * dir||
        diff = pos - ray_origin[np.newaxis, :]  # [M, 3]
        proj = np.sum(diff * ray_dir[np.newaxis, :], axis=-1, keepdims=True)

        # Only consider points in front of camera
        front_mask = proj[:, 0] > 0
        if not np.any(front_mask):
            self.selected_index = None
            return None

        closest_pt = ray_origin + proj * ray_dir[np.newaxis, :]
        dist = np.linalg.norm(pos - closest_pt, axis=-1)

        # Weight by scale (larger Gaussians are easier to hit)
        avg_scale = np.mean(scales[visible_idx], axis=-1)  # [M]
        effective_dist = dist / np.maximum(avg_scale, 1e-8)

        # Mask out behind-camera
        effective_dist[~front_mask] = np.inf

        best_local = np.argmin(effective_dist)
        # Threshold: don't select if too far
        if effective_dist[best_local] > 50.0:
            self.selected_index = None
            return None

        self.selected_index = int(visible_idx[best_local])
        return self.selected_index

    def get_properties(self, index: int, data: Checkpoint4DData,
                       current_time: float) -> dict:
        """Get full property dict for a Gaussian at current time."""
        i = index
        dt = current_time - data.times[i, 0]
        s = max(np.exp(data.durations[i, 0]), 0.02)
        temporal_opa = float(np.exp(-0.5 * (dt / s) ** 2))

        # Compute current position
        pos = data.means[i].copy()
        pos += data.velocities[i] * dt
        if data.motion_order >= 2 and data.accels is not None:
            pos += 0.5 * data.accels[i] * dt ** 2
        if data.motion_order >= 3 and data.jerks is not None:
            pos += (1/6) * data.jerks[i] * dt ** 3
        if data.motion_order >= 4 and data.snaps is not None:
            pos += (1/24) * data.snaps[i] * dt ** 4

        base_opa = float(1.0 / (1.0 + np.exp(-data.opacities[i, 0])))

        props = {
            "index": i,
            "canonical_pos": data.means[i].tolist(),
            "current_pos": pos.tolist(),
            "velocity": data.velocities[i].tolist(),
            "canonical_time": float(data.times[i, 0]),
            "duration": float(s),
            "duration_log": float(data.durations[i, 0]),
            "base_opacity": base_opa,
            "temporal_opacity": temporal_opa,
            "final_opacity": base_opa * temporal_opa,
            "motion_order": data.motion_order,
        }

        if data.motion_order >= 2 and data.accels is not None:
            props["acceleration"] = data.accels[i].tolist()
        if data.motion_order >= 3 and data.jerks is not None:
            props["jerk"] = data.jerks[i].tolist()
        if data.motion_order >= 4 and data.snaps is not None:
            props["snap"] = data.snaps[i].tolist()

        if hasattr(data, 'grad_accum') and data.grad_accum is not None:
            props["grad_accum"] = float(data.grad_accum[i])

        return props

    def deselect(self):
        self.selected_index = None
