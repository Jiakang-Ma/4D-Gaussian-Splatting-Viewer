"""Unit tests for loader_4d.Checkpoint4DData and load_checkpoint."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest
import torch

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loader_4d import Checkpoint4DData, load_checkpoint


# ======================================================================
# Helpers
# ======================================================================

def _make_splats(n: int = 50, motion_order: int = 1) -> OrderedDict:
    """Build a minimal valid splats state_dict."""
    splats = OrderedDict(
        means=torch.randn(n, 3),
        scales=torch.randn(n, 3),          # log space
        quats=torch.randn(n, 4),
        opacities=torch.randn(n),           # logit space
        sh0=torch.randn(n, 1, 3),
        shN=torch.randn(n, 15, 3),          # sh_degree=3 → (4²-1)=15
        times=torch.rand(n, 1),
        durations=torch.randn(n, 1),        # log space
        velocities=torch.randn(n, 3),
    )
    if motion_order >= 2:
        splats["accels"] = torch.randn(n, 3)
    if motion_order >= 3:
        splats["jerks"] = torch.randn(n, 3)
    if motion_order >= 4:
        splats["snaps"] = torch.randn(n, 3)
    return splats


def _save_ckpt(path: str, splats: OrderedDict, step: int = 1000) -> None:
    torch.save({"step": step, "splats": splats, "optimizers": {}}, path)


# ======================================================================
# Checkpoint4DData dataclass tests
# ======================================================================

class TestCheckpoint4DData:
    def test_default_optional_fields(self):
        data = Checkpoint4DData(
            means=np.zeros((1, 3)),
            scales=np.zeros((1, 3)),
            quats=np.zeros((1, 4)),
            opacities=np.zeros((1, 1)),
            sh=np.zeros((1, 3)),
            times=np.zeros((1, 1)),
            durations=np.zeros((1, 1)),
            velocities=np.zeros((1, 3)),
        )
        assert data.accels is None
        assert data.jerks is None
        assert data.snaps is None
        assert data.motion_order == 1
        assert data.train_step == 0


# ======================================================================
# load_checkpoint tests
# ======================================================================

class TestLoadCheckpoint:
    def test_basic_load_order1(self, tmp_path):
        """Load a motion_order=1 checkpoint and verify fields."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        splats = _make_splats(30, motion_order=1)
        _save_ckpt(ckpt_path, splats, step=5000)

        data = load_checkpoint(ckpt_path)

        assert isinstance(data, Checkpoint4DData)
        assert data.n_gaussians == 30
        assert data.train_step == 5000
        assert data.motion_order == 1
        assert data.means.shape == (30, 3)
        assert data.scales.shape == (30, 3)
        assert data.quats.shape == (30, 4)
        assert data.opacities.shape == (30, 1)
        assert data.times.shape == (30, 1)
        assert data.durations.shape == (30, 1)
        assert data.velocities.shape == (30, 3)
        # sh0[30,1,3] + shN[30,15,3] → [30, 48]
        assert data.sh.shape == (30, 48)
        assert data.accels is None
        assert data.jerks is None
        assert data.snaps is None

    def test_motion_order_detection(self, tmp_path):
        """motion_order is detected from higher-order keys."""
        for order in [1, 2, 3, 4]:
            ckpt_path = str(tmp_path / f"ckpt_o{order}.pt")
            _save_ckpt(ckpt_path, _make_splats(20, motion_order=order))
            data = load_checkpoint(ckpt_path)
            assert data.motion_order == order

    def test_higher_order_shapes(self, tmp_path):
        """accels/jerks/snaps have correct shapes when present."""
        ckpt_path = str(tmp_path / "ckpt4.pt")
        _save_ckpt(ckpt_path, _make_splats(40, motion_order=4))

        data = load_checkpoint(ckpt_path)
        assert data.accels is not None and data.accels.shape == (40, 3)
        assert data.jerks is not None and data.jerks.shape == (40, 3)
        assert data.snaps is not None and data.snaps.shape == (40, 3)

    def test_scales_exp_activated(self, tmp_path):
        """Scales should be exp-activated (all positive)."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        splats = _make_splats(10)
        # Set known log-scales
        splats["scales"] = torch.tensor([[0.0, 1.0, -1.0]] * 10)
        _save_ckpt(ckpt_path, splats)

        data = load_checkpoint(ckpt_path)
        expected = np.exp(np.array([[0.0, 1.0, -1.0]] * 10, dtype=np.float32))
        np.testing.assert_allclose(data.scales, expected, rtol=1e-5)

    def test_quats_normalized(self, tmp_path):
        """Quaternions should be L2-normalized."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        _save_ckpt(ckpt_path, _make_splats(20))

        data = load_checkpoint(ckpt_path)
        norms = np.linalg.norm(data.quats, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_sh_concatenation(self, tmp_path):
        """SH is sh0 + shN concatenated and flattened."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        splats = _make_splats(5)
        # Set known SH values
        splats["sh0"] = torch.ones(5, 1, 3) * 0.5
        splats["shN"] = torch.ones(5, 15, 3) * 0.1
        _save_ckpt(ckpt_path, splats)

        data = load_checkpoint(ckpt_path)
        # First 3 values should be 0.5 (sh0), rest should be 0.1 (shN)
        assert data.sh.shape == (5, 48)
        np.testing.assert_allclose(data.sh[0, :3], 0.5, atol=1e-6)
        np.testing.assert_allclose(data.sh[0, 3:], 0.1, atol=1e-6)

    def test_time_min_max(self, tmp_path):
        """time_min and time_max match the actual times tensor."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        splats = _make_splats(10)
        splats["times"] = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9],
                                         [0.2], [0.4], [0.6], [0.8], [0.0]])
        _save_ckpt(ckpt_path, splats)

        data = load_checkpoint(ckpt_path)
        assert abs(data.time_min - 0.0) < 1e-6
        assert abs(data.time_max - 0.9) < 1e-6

    def test_missing_splats_key(self, tmp_path):
        """Raise ValueError when checkpoint has no 'splats' key."""
        ckpt_path = str(tmp_path / "bad.pt")
        torch.save({"step": 0}, ckpt_path)

        with pytest.raises(ValueError, match="missing 'splats' key"):
            load_checkpoint(ckpt_path)

    def test_step_defaults_to_zero(self, tmp_path):
        """When 'step' key is absent, default to 0."""
        ckpt_path = str(tmp_path / "nostep.pt")
        torch.save({"splats": _make_splats(5)}, ckpt_path)

        data = load_checkpoint(ckpt_path)
        assert data.train_step == 0

    def test_all_arrays_float32(self, tmp_path):
        """All numpy arrays should be float32."""
        ckpt_path = str(tmp_path / "ckpt.pt")
        _save_ckpt(ckpt_path, _make_splats(10, motion_order=4))

        data = load_checkpoint(ckpt_path)
        for name in ["means", "scales", "quats", "opacities", "sh",
                      "times", "durations", "velocities",
                      "accels", "jerks", "snaps"]:
            arr = getattr(data, name)
            if arr is not None:
                assert arr.dtype == np.float32, f"{name} is {arr.dtype}"

    def test_file_not_found(self):
        """Raise error for non-existent path."""
        with pytest.raises(Exception):
            load_checkpoint("/nonexistent/path/ckpt.pt")
