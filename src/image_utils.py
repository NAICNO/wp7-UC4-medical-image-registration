"""
Pure numpy/scipy utility functions for 3D medical image processing.

These functions are extracted from the project notebooks and isolate
the reusable computation logic that does NOT depend on heavy libraries
(nibabel, antspyx, hd-bet, monai, SimpleITK).

Notebook origins:
  - notebooks/3D-image-registration.ipynb
      brats_ants_mni152betafter_registration() uses:
        * np.ones((3,3,3))                       -> compute_neighbor_structure()
        * np.where(arr > 0, 1., 0.)              -> compute_binary_mask()
        * ndimage.binary_fill_holes(...)          -> fill_holes_3d()
        * element-wise mask multiplication       -> apply_binary_mask()

  - notebooks/BraTS-sri24-AtlasProcessBrainExtract.ipynb
      get_ImageBasicInfo() prints shape/zoom/orientation info.
      The origin correction block:
        * copy.deepcopy(arr).astype(np.float32)  -> clip_volume() / normalize_volume()

All functions treat input arrays as immutable: they return new arrays
without modifying the originals.
"""

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SUBJECT_KEYS = ("SubjectID", "t1wPath", "t1cwPath", "t2wPath", "flairPath")


# ---------------------------------------------------------------------------
# Mask operations  (core logic from 3D-image-registration.ipynb)
# ---------------------------------------------------------------------------


def apply_binary_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to a 3D volume.

    Mirrors the element-wise multiplication used in the registration
    pipeline: ``registered_image * brain_mask``.

    Parameters
    ----------
    volume : np.ndarray
        3-D array of voxel intensities (float or int).
    mask : np.ndarray
        3-D binary array with the same shape as *volume*.
        Non-zero values are treated as brain region.

    Returns
    -------
    np.ndarray
        New array with voxels outside the mask set to zero.

    Raises
    ------
    ValueError
        If *volume* and *mask* shapes differ, or input is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    if volume.shape != mask.shape:
        raise ValueError(
            f"volume shape {volume.shape} does not match mask shape {mask.shape}"
        )
    binary = (mask != 0).astype(volume.dtype)
    return volume * binary


def compute_binary_mask(volume: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Convert a volume into a binary mask.

    Mirrors ``np.where(arr > threshold, 1., 0.)`` used when preparing
    the T1Gd brain ROI mask in the registration notebook.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D array.
    threshold : float
        Voxels strictly above this value become 1; others become 0.

    Returns
    -------
    np.ndarray
        Float32 array of the same shape containing only 0.0 and 1.0.

    Raises
    ------
    ValueError
        If *volume* is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    return np.where(volume > threshold, 1.0, 0.0).astype(np.float32)


def fill_holes_3d(
    mask: np.ndarray,
    structure: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fill holes in a 3-D binary mask using ``scipy.ndimage.binary_fill_holes``.

    Mirrors:
        ``ndimage.binary_fill_holes(mask_array, structure=neighb_struct)``

    Parameters
    ----------
    mask : np.ndarray
        3-D binary (or boolean) array.
    structure : np.ndarray, optional
        Structuring element passed to ``binary_fill_holes``.
        Defaults to a 3x3x3 array of ones (same default as the notebook).

    Returns
    -------
    np.ndarray
        Float32 array with holes filled.

    Raises
    ------
    ValueError
        If *mask* is not 3-D.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    if structure is None:
        structure = compute_neighbor_structure()
    binary = mask.astype(bool)
    filled = ndimage.binary_fill_holes(binary, structure=structure)
    return filled.astype(np.float32)


# ---------------------------------------------------------------------------
# Structuring element  (from 3D-image-registration.ipynb)
# ---------------------------------------------------------------------------


def compute_neighbor_structure(size: int = 3) -> np.ndarray:
    """
    Return a cubic structuring element of ones.

    Mirrors ``np.ones((3, 3, 3))`` used in the registration pipeline as
    the structuring element for morphological hole-filling.

    Parameters
    ----------
    size : int
        Side length of the cube. Default 3 matches the notebook.

    Returns
    -------
    np.ndarray
        Float64 array of shape (size, size, size) filled with 1.0.

    Raises
    ------
    ValueError
        If *size* is not a positive integer.
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError(f"size must be a positive integer, got {size!r}")
    return np.ones((size, size, size))


# ---------------------------------------------------------------------------
# Intensity normalization / clipping
# ---------------------------------------------------------------------------


def normalize_volume(
    volume: np.ndarray,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> np.ndarray:
    """
    Linearly rescale voxel intensities to [new_min, new_max].

    Useful for normalizing MRI intensities before visualization or
    feeding to downstream algorithms.

    Parameters
    ----------
    volume : np.ndarray
        3-D input array.
    new_min, new_max : float
        Target intensity range.

    Returns
    -------
    np.ndarray
        Float64 array of the same shape with values in [new_min, new_max].
        If the volume has zero range (all values identical), returns an
        array filled with *new_min*.

    Raises
    ------
    ValueError
        If *volume* is not 3-D or *new_min* >= *new_max*.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if new_min >= new_max:
        raise ValueError(f"new_min ({new_min}) must be less than new_max ({new_max})")
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax == vmin:
        return np.full(volume.shape, new_min, dtype=np.float64)
    return (volume.astype(np.float64) - vmin) / (vmax - vmin) * (new_max - new_min) + new_min


def clip_volume(
    volume: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """
    Clip voxel intensities to [low, high].

    Parameters
    ----------
    volume : np.ndarray
        3-D input array.
    low, high : float
        Clip bounds.

    Returns
    -------
    np.ndarray
        New array of the same shape with values clipped.

    Raises
    ------
    ValueError
        If *volume* is not 3-D or *low* > *high*.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return np.clip(volume, low, high)


def threshold_volume(volume: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarize a volume: voxels > threshold become 1, others 0.

    Parameters
    ----------
    volume : np.ndarray
        3-D input array.
    threshold : float
        Threshold value.

    Returns
    -------
    np.ndarray
        Float32 binary array of the same shape.

    Raises
    ------
    ValueError
        If *volume* is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    return (volume > threshold).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_volume_statistics(volume: np.ndarray) -> Dict[str, float]:
    """
    Compute basic descriptive statistics of a 3-D volume.

    Parameters
    ----------
    volume : np.ndarray
        3-D array.

    Returns
    -------
    dict with keys: min, max, mean, std, nonzero_count, total_voxels.

    Raises
    ------
    ValueError
        If *volume* is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    return {
        "min": float(volume.min()),
        "max": float(volume.max()),
        "mean": float(volume.mean()),
        "std": float(volume.std()),
        "nonzero_count": int(np.count_nonzero(volume)),
        "total_voxels": int(volume.size),
    }


def count_nonzero_voxels(volume: np.ndarray) -> int:
    """
    Return the number of non-zero voxels in a 3-D array.

    Parameters
    ----------
    volume : np.ndarray
        3-D array.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If *volume* is not 3-D.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    return int(np.count_nonzero(volume))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_volume_shape(volume: np.ndarray) -> bool:
    """
    Return True if *volume* is a non-empty 3-D numpy array.

    Parameters
    ----------
    volume : np.ndarray
        Candidate volume.

    Returns
    -------
    bool
    """
    return isinstance(volume, np.ndarray) and volume.ndim == 3 and volume.size > 0


def validate_subject_dict(subject_dict: dict) -> Tuple[bool, str]:
    """
    Validate that a subject dictionary has all required keys.

    Mirrors the structure expected by
    ``brats_ants_mni152betafter_registration()``:
        SubjectID, t1wPath, t1cwPath, t2wPath, flairPath.

    Parameters
    ----------
    subject_dict : dict
        Dictionary to validate.

    Returns
    -------
    (bool, str)
        (True, "") if valid; (False, <reason>) otherwise.
    """
    if not isinstance(subject_dict, dict):
        return False, "subject_dict must be a dict"
    missing = [k for k in REQUIRED_SUBJECT_KEYS if k not in subject_dict]
    if missing:
        return False, f"Missing keys: {missing}"
    if not isinstance(subject_dict.get("SubjectID"), str):
        return False, "SubjectID must be a string"
    if not subject_dict["SubjectID"].strip():
        return False, "SubjectID must not be empty"
    return True, ""


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def build_subject_output_path(save_dir: str, subject_id: str) -> str:
    """
    Construct the expected output directory path for a given subject.

    Mirrors ``os.path.join(save_dir, aSubId)`` in the registration function.

    Parameters
    ----------
    save_dir : str
        Root results directory.
    subject_id : str
        Subject identifier (e.g. "EGD-0117").

    Returns
    -------
    str
        Joined path.

    Raises
    ------
    ValueError
        If either argument is empty.
    """
    if not save_dir or not save_dir.strip():
        raise ValueError("save_dir must not be empty")
    if not subject_id or not subject_id.strip():
        raise ValueError("subject_id must not be empty")
    return os.path.join(save_dir, subject_id)


# ---------------------------------------------------------------------------
# Spatial operations
# ---------------------------------------------------------------------------


def compute_bounding_box(
    mask: np.ndarray,
) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
    """
    Compute the tight axis-aligned bounding box of non-zero voxels.

    Parameters
    ----------
    mask : np.ndarray
        3-D binary (or boolean) array.

    Returns
    -------
    ((x_min, x_max), (y_min, y_max), (z_min, z_max)) or None if mask is empty.

    Raises
    ------
    ValueError
        If *mask* is not 3-D.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    coords = np.argwhere(mask != 0)
    if coords.size == 0:
        return None
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    return (
        (int(mins[0]), int(maxs[0])),
        (int(mins[1]), int(maxs[1])),
        (int(mins[2]), int(maxs[2])),
    )


def extract_brain_roi(
    volume: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Apply a brain mask and return the masked volume.

    This combines ``compute_binary_mask`` and ``apply_binary_mask`` in
    a single convenience call, mirroring the pattern used after
    registration in the notebook:
        ``registered * atlas_mask``

    Parameters
    ----------
    volume : np.ndarray
        3-D voxel array.
    mask : np.ndarray
        3-D brain-region mask.

    Returns
    -------
    np.ndarray
        Volume with non-brain voxels zeroed.
    """
    binary_mask = compute_binary_mask(mask, threshold=0.0)
    return apply_binary_mask(volume, binary_mask)


def resample_volume_nearest(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Resample a 3-D volume to *target_shape* using nearest-neighbour interpolation.

    Uses ``scipy.ndimage.zoom``.

    Parameters
    ----------
    volume : np.ndarray
        3-D source array.
    target_shape : tuple of ints
        Desired (D, H, W) output shape.

    Returns
    -------
    np.ndarray
        Resampled array of shape *target_shape*.

    Raises
    ------
    ValueError
        If *volume* is not 3-D or *target_shape* has wrong length.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if len(target_shape) != 3:
        raise ValueError(f"target_shape must have 3 elements, got {len(target_shape)}")
    zoom_factors = tuple(t / s for t, s in zip(target_shape, volume.shape))
    return ndimage.zoom(volume, zoom_factors, order=0)


def pad_volume_to_shape(
    volume: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Zero-pad a 3-D volume to reach *target_shape*.

    Padding is added symmetrically; if odd, extra voxels go at the end.

    Parameters
    ----------
    volume : np.ndarray
        3-D source array.
    target_shape : tuple of ints
        Desired output shape (must be >= volume.shape in every dimension).
    pad_value : float
        Fill value for padded voxels.

    Returns
    -------
    np.ndarray
        Padded array of shape *target_shape*.

    Raises
    ------
    ValueError
        If *volume* is not 3-D, *target_shape* is wrong length, or any
        target dimension is smaller than the source dimension.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3-D, got shape {volume.shape}")
    if len(target_shape) != 3:
        raise ValueError(f"target_shape must have 3 elements, got {len(target_shape)}")
    for dim, (src, tgt) in enumerate(zip(volume.shape, target_shape)):
        if tgt < src:
            raise ValueError(
                f"target_shape[{dim}]={tgt} is smaller than volume.shape[{dim}]={src}"
            )
    pad_width = []
    for src, tgt in zip(volume.shape, target_shape):
        total = tgt - src
        before = total // 2
        after = total - before
        pad_width.append((before, after))
    return np.pad(volume, pad_width, mode="constant", constant_values=pad_value)


def crop_volume_to_bounding_box(
    volume: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Crop a volume to the tight bounding box of *mask*'s non-zero region.

    Parameters
    ----------
    volume : np.ndarray
        3-D voxel array.
    mask : np.ndarray
        3-D binary mask.

    Returns
    -------
    np.ndarray
        Cropped sub-volume.

    Raises
    ------
    ValueError
        If shapes differ, inputs are not 3-D, or mask is all zeros.
    """
    if volume.shape != mask.shape:
        raise ValueError(
            f"volume shape {volume.shape} does not match mask shape {mask.shape}"
        )
    bbox = compute_bounding_box(mask)
    if bbox is None:
        raise ValueError("mask is all zeros; cannot compute bounding box")
    (x0, x1), (y0, y1), (z0, z1) = bbox
    return volume[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1]


def compute_center_of_mass(mask: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """
    Compute the centre of mass of non-zero voxels.

    Parameters
    ----------
    mask : np.ndarray
        3-D array.

    Returns
    -------
    (x, y, z) float tuple, or None if mask is empty.

    Raises
    ------
    ValueError
        If *mask* is not 3-D.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3-D, got shape {mask.shape}")
    result = ndimage.center_of_mass(mask)
    if any(np.isnan(v) for v in result):
        return None
    return tuple(float(v) for v in result)


# ---------------------------------------------------------------------------
# Overlap / similarity metrics
# ---------------------------------------------------------------------------


def compute_dice_coefficient(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Compute the Dice similarity coefficient between two binary masks.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        3-D binary arrays of the same shape.

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both masks are empty.

    Raises
    ------
    ValueError
        If shapes differ or inputs are not 3-D.
    """
    if mask_a.ndim != 3 or mask_b.ndim != 3:
        raise ValueError("Both masks must be 3-D")
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"mask_a shape {mask_a.shape} != mask_b shape {mask_b.shape}"
        )
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = int(np.sum(a & b))
    denom = int(np.sum(a)) + int(np.sum(b))
    if denom == 0:
        return 1.0
    return 2.0 * intersection / denom


def compute_volume_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """
    Compute the Jaccard (Intersection over Union) overlap between two binary masks.

    IoU = |A ∩ B| / |A ∪ B|

    Parameters
    ----------
    mask_a, mask_b : np.ndarray
        3-D binary arrays of the same shape.

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both masks are empty.

    Raises
    ------
    ValueError
        If shapes differ or inputs are not 3-D.
    """
    if mask_a.ndim != 3 or mask_b.ndim != 3:
        raise ValueError("Both masks must be 3-D")
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"mask_a shape {mask_a.shape} != mask_b shape {mask_b.shape}"
        )
    a = (mask_a > 0).astype(np.uint8)
    b = (mask_b > 0).astype(np.uint8)
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))
    if union == 0:
        return 1.0
    return float(intersection) / float(union)
