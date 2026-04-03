"""
4D Gaussian Splatting checkpoint loader.

Parses .pt checkpoint files produced by the FreeTimeGS trainer,
extracting splats parameters and converting them to numpy arrays
suitable for OpenGL rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch


@dataclass
class Checkpoint4DData:
    """Structured 4D Gaussian data parsed from a checkpoint."""

    # Spatial parameters
    means: np.ndarray
    scales: np.ndarray
    quats: np.ndarray
    opacities: np.ndarray
    sh: np.ndarray

    # Temporal parameters
    times: np.ndarray
    durations: np.ndarray
    velocities: np.ndarray

    # Optional higher-order motion
    accels: Optional[np.ndarray] = None
    jerks: Optional[np.ndarray] = None
    snaps: Optional[np.ndarray] = None

    # Gradient data (optional, from training)
    grad_accum: Optional[np.ndarray] = None

    # Metadata
    motion_order: int = 1
    train_step: int = 0
    n_gaussians: int = 0
    time_min: float = 0.0
    time_max: float = 1.0


def _to_numpy(t) -> np.ndarray:
    if isinstance(t, torch.nn.Parameter):
        t = t.data
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().float().numpy()
    return np.asarray(t, dtype=np.float32)


def load_checkpoint(path: str) -> Checkpoint4DData:
    """Load a .pt checkpoint and return structured 4D Gaussian data."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if "splats" not in ckpt:
        raise ValueError(f"Checkpoint missing 'splats' key: {path}")

    splats = ckpt["splats"]

    means = _to_numpy(splats["means"])

    raw_scales = splats["scales"]
    if isinstance(raw_scales, torch.nn.Parameter):
        raw_scales = raw_scales.data
    scales = _to_numpy(torch.exp(raw_scales.detach().cpu().float()))

    raw_quats = splats["quats"]
    if isinstance(raw_quats, torch.nn.Parameter):
        raw_quats = raw_quats.data
    quats_t = raw_quats.detach().cpu().float()
    quats = _to_numpy(quats_t / quats_t.norm(dim=-1, keepdim=True))

    raw_opa = splats["opacities"]
    if isinstance(raw_opa, torch.nn.Parameter):
        raw_opa = raw_opa.data
    opa_np = _to_numpy(raw_opa.detach().cpu().float())
    opacities = opa_np[:, np.newaxis] if opa_np.ndim == 1 else opa_np

    sh0 = _to_numpy(splats["sh0"])
    shN = _to_numpy(splats["shN"])
    sh0_flat = sh0.reshape(sh0.shape[0], -1)
    shN_flat = shN.reshape(shN.shape[0], -1)
    sh = np.concatenate([sh0_flat, shN_flat], axis=-1).astype(np.float32)

    times = _to_numpy(splats["times"])
    durations = _to_numpy(splats["durations"])
    velocities = _to_numpy(splats["velocities"])

    motion_order = 1
    accels = jerks = snaps = None
    if "accels" in splats:
        accels = _to_numpy(splats["accels"])
        motion_order = 2
    if "jerks" in splats:
        jerks = _to_numpy(splats["jerks"])
        motion_order = 3
    if "snaps" in splats:
        snaps = _to_numpy(splats["snaps"])
        motion_order = 4

    # Gradient accumulator (optional)
    grad_accum = None
    if "grad_accum" in ckpt and ckpt["grad_accum"] is not None:
        ga = ckpt["grad_accum"]
        if isinstance(ga, torch.Tensor):
            grad_accum = ga.detach().cpu().float().numpy()
        else:
            grad_accum = np.asarray(ga, dtype=np.float32)

    n_gaussians = means.shape[0]

    return Checkpoint4DData(
        means=means,
        scales=scales,
        quats=quats,
        opacities=opacities,
        sh=sh,
        times=times,
        durations=durations,
        velocities=velocities,
        accels=accels,
        jerks=jerks,
        snaps=snaps,
        grad_accum=grad_accum,
        motion_order=motion_order,
        train_step=int(ckpt.get("step", 0)),
        n_gaussians=n_gaussians,
        time_min=float(times.min()),
        time_max=float(times.max()),
    )
