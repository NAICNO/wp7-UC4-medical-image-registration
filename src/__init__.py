"""
src package for 3D Medical Image Registration Segmentation.

Provides pure numpy/scipy utility functions extracted from the project
notebooks, testable without heavy dependencies (nibabel, antspyx, etc.).
"""

from src.image_utils import (
    apply_binary_mask,
    normalize_volume,
    compute_binary_mask,
    fill_holes_3d,
    clip_volume,
    compute_volume_statistics,
    validate_volume_shape,
    validate_subject_dict,
    build_subject_output_path,
    compute_neighbor_structure,
    threshold_volume,
    count_nonzero_voxels,
    compute_bounding_box,
    extract_brain_roi,
    resample_volume_nearest,
    pad_volume_to_shape,
    crop_volume_to_bounding_box,
    compute_center_of_mass,
    compute_dice_coefficient,
    compute_volume_overlap,
)

__all__ = [
    "apply_binary_mask",
    "normalize_volume",
    "compute_binary_mask",
    "fill_holes_3d",
    "clip_volume",
    "compute_volume_statistics",
    "validate_volume_shape",
    "validate_subject_dict",
    "build_subject_output_path",
    "compute_neighbor_structure",
    "threshold_volume",
    "count_nonzero_voxels",
    "compute_bounding_box",
    "extract_brain_roi",
    "resample_volume_nearest",
    "pad_volume_to_shape",
    "crop_volume_to_bounding_box",
    "compute_center_of_mass",
    "compute_dice_coefficient",
    "compute_volume_overlap",
]
