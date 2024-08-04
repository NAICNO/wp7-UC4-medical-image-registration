"""
Pytest fixtures shared across the entire test suite.

All 3D arrays are small synthetic volumes (8x8x8) so tests run fast
without any I/O or heavy dependencies.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

import os as _os
PROJECT_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))

NOTEBOOKS_DIR = _os.path.join(PROJECT_ROOT, "notebooks")
DATA_DIR = _os.path.join(PROJECT_ROOT, "data")
ASSETS_DIR = _os.path.join(PROJECT_ROOT, "assets")
RESULTS_DIR = _os.path.join(PROJECT_ROOT, "results")
SRC_DIR = _os.path.join(PROJECT_ROOT, "src")

NOTEBOOK_FILENAMES = [
    "3D-image-registration.ipynb",
    "BraTS-sri24-AtlasProcessBrainExtract.ipynb",
    "antsPyRegistrationBasic.ipynb",
]

# ---------------------------------------------------------------------------
# Synthetic 3-D volumes  (8 x 8 x 8)
# ---------------------------------------------------------------------------

VOLUME_SHAPE = (8, 8, 8)


@pytest.fixture
def zero_volume():
    """3-D array of all zeros."""
    return np.zeros(VOLUME_SHAPE, dtype=np.float32)


@pytest.fixture
def ones_volume():
    """3-D array of all ones."""
    return np.ones(VOLUME_SHAPE, dtype=np.float32)


@pytest.fixture
def rng_volume():
    """3-D array of uniform random floats in [0, 1)."""
    rng = np.random.default_rng(seed=42)
    return rng.random(VOLUME_SHAPE).astype(np.float32)


@pytest.fixture
def integer_volume():
    """3-D array with integer voxels in [0, 255]."""
    rng = np.random.default_rng(seed=7)
    return rng.integers(0, 256, size=VOLUME_SHAPE).astype(np.float32)


@pytest.fixture
def binary_mask():
    """3-D binary mask: a 4x4x4 cube of ones inside an 8x8x8 zeros array."""
    mask = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    mask[2:6, 2:6, 2:6] = 1.0
    return mask


@pytest.fixture
def binary_mask_with_holes():
    """Binary mask that has an interior hole to test fill_holes_3d."""
    mask = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    mask[1:7, 1:7, 1:7] = 1.0
    mask[3:5, 3:5, 3:5] = 0.0  # punch a hole in the middle
    return mask


@pytest.fixture
def gradient_volume():
    """3-D volume where values increase along the x-axis."""
    arr = np.zeros(VOLUME_SHAPE, dtype=np.float64)
    for i in range(VOLUME_SHAPE[0]):
        arr[i, :, :] = float(i)
    return arr


@pytest.fixture
def uniform_volume():
    """3-D volume where every voxel equals 5.0 (zero range)."""
    return np.full(VOLUME_SHAPE, 5.0, dtype=np.float64)


@pytest.fixture
def small_volume_5x5x5():
    """A 5x5x5 volume for boundary / different-shape tests."""
    rng = np.random.default_rng(seed=99)
    return rng.random((5, 5, 5)).astype(np.float32)


@pytest.fixture
def large_brain_like_volume():
    """Synthetic 'brain-like' volume: 16x16x16 with a spherical blob."""
    shape = (16, 16, 16)
    centre = np.array([8, 8, 8], dtype=float)
    coords = np.indices(shape).reshape(3, -1).T
    dist = np.linalg.norm(coords - centre, axis=1).reshape(shape)
    vol = np.where(dist < 6, (6 - dist) * 10.0, 0.0).astype(np.float32)
    return vol


@pytest.fixture
def sphere_mask():
    """16x16x16 spherical binary mask centred at (8,8,8)."""
    shape = (16, 16, 16)
    centre = np.array([8, 8, 8], dtype=float)
    coords = np.indices(shape).reshape(3, -1).T
    dist = np.linalg.norm(coords - centre, axis=1).reshape(shape)
    return (dist < 6).astype(np.float32)


@pytest.fixture
def valid_subject_dict():
    """A correctly structured subject dictionary for validation tests."""
    return {
        "SubjectID": "EGD-0117",
        "t1wPath": "../data/EGD-0117/T1.nii.gz",
        "t1cwPath": "../data/EGD-0117/T1GD.nii.gz",
        "t2wPath": "../data/EGD-0117/T2.nii.gz",
        "flairPath": "../data/EGD-0117/FLAIR.nii.gz",
    }
