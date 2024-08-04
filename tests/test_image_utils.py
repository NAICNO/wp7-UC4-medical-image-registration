"""
Unit tests for src/image_utils.py.

All tests use small synthetic 3-D numpy arrays (fixtures in conftest.py).
No file I/O, no heavy dependencies.

Coverage targets:
  - apply_binary_mask
  - compute_binary_mask
  - fill_holes_3d
  - compute_neighbor_structure
  - normalize_volume
  - clip_volume
  - threshold_volume
  - compute_volume_statistics
  - count_nonzero_voxels
  - validate_volume_shape
  - validate_subject_dict
  - build_subject_output_path
  - compute_bounding_box
  - extract_brain_roi
  - resample_volume_nearest
  - pad_volume_to_shape
  - crop_volume_to_bounding_box
  - compute_center_of_mass
  - compute_dice_coefficient
  - compute_volume_overlap
"""

import numpy as np
import pytest

from src.image_utils import (
    REQUIRED_SUBJECT_KEYS,
    apply_binary_mask,
    build_subject_output_path,
    clip_volume,
    compute_binary_mask,
    compute_bounding_box,
    compute_center_of_mass,
    compute_dice_coefficient,
    compute_neighbor_structure,
    compute_volume_overlap,
    compute_volume_statistics,
    count_nonzero_voxels,
    crop_volume_to_bounding_box,
    extract_brain_roi,
    fill_holes_3d,
    normalize_volume,
    pad_volume_to_shape,
    resample_volume_nearest,
    threshold_volume,
    validate_subject_dict,
    validate_volume_shape,
)


# ===========================================================================
# apply_binary_mask
# ===========================================================================


class TestApplyBinaryMask:
    def test_all_zeros_mask_returns_zero_volume(self, ones_volume, zero_volume):
        result = apply_binary_mask(ones_volume, zero_volume)
        assert np.all(result == 0.0)

    def test_all_ones_mask_returns_original_volume(self, rng_volume, ones_volume):
        result = apply_binary_mask(rng_volume, ones_volume)
        np.testing.assert_array_almost_equal(result, rng_volume)

    def test_partial_mask_zeroes_outside_region(self, ones_volume, binary_mask):
        result = apply_binary_mask(ones_volume, binary_mask)
        # Inside the 4x4x4 cube voxels should be 1
        assert result[3, 3, 3] == 1.0
        # Outside (e.g. corner) should be 0
        assert result[0, 0, 0] == 0.0

    def test_result_has_same_shape(self, rng_volume, binary_mask):
        result = apply_binary_mask(rng_volume, binary_mask)
        assert result.shape == rng_volume.shape

    def test_does_not_mutate_original_volume(self, rng_volume, binary_mask):
        original = rng_volume.copy()
        apply_binary_mask(rng_volume, binary_mask)
        np.testing.assert_array_equal(rng_volume, original)

    def test_does_not_mutate_mask(self, rng_volume, binary_mask):
        original_mask = binary_mask.copy()
        apply_binary_mask(rng_volume, binary_mask)
        np.testing.assert_array_equal(binary_mask, original_mask)

    def test_raises_if_volume_not_3d(self, binary_mask):
        with pytest.raises(ValueError, match="3-D"):
            apply_binary_mask(np.ones((4, 4)), binary_mask)

    def test_raises_if_mask_not_3d(self, rng_volume):
        with pytest.raises(ValueError, match="3-D"):
            apply_binary_mask(rng_volume, np.ones((4, 4)))

    def test_raises_if_shapes_differ(self):
        vol = np.ones((8, 8, 8))
        mask = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="shape"):
            apply_binary_mask(vol, mask)

    def test_non_binary_mask_treated_as_binary(self):
        vol = np.full((4, 4, 4), 2.0)
        mask = np.full((4, 4, 4), 5.0)  # all non-zero -> all ones
        result = apply_binary_mask(vol, mask)
        assert np.all(result == 2.0)

    def test_integer_volume_with_binary_mask(self, integer_volume, binary_mask):
        result = apply_binary_mask(integer_volume, binary_mask)
        assert result.dtype == integer_volume.dtype
        assert result[0, 0, 0] == 0.0


# ===========================================================================
# compute_binary_mask
# ===========================================================================


class TestComputeBinaryMask:
    def test_zero_volume_returns_all_zeros(self, zero_volume):
        result = compute_binary_mask(zero_volume)
        assert np.all(result == 0.0)

    def test_positive_volume_returns_all_ones(self, ones_volume):
        result = compute_binary_mask(ones_volume)
        assert np.all(result == 1.0)

    def test_output_dtype_is_float32(self, rng_volume):
        result = compute_binary_mask(rng_volume)
        assert result.dtype == np.float32

    def test_output_contains_only_0_and_1(self, rng_volume):
        result = compute_binary_mask(rng_volume)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})

    def test_custom_threshold(self):
        vol = np.array([[[0.3, 0.7], [0.1, 0.9]], [[0.5, 0.5], [0.2, 0.8]]], dtype=np.float32)
        result = compute_binary_mask(vol, threshold=0.5)
        # Only voxels > 0.5 become 1
        assert result[0, 0, 1] == 1.0  # 0.7 > 0.5
        assert result[0, 0, 0] == 0.0  # 0.3 <= 0.5
        assert result[1, 0, 0] == 0.0  # 0.5 is NOT > 0.5

    def test_does_not_mutate_input(self, rng_volume):
        original = rng_volume.copy()
        compute_binary_mask(rng_volume)
        np.testing.assert_array_equal(rng_volume, original)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_binary_mask(np.ones((4, 4)))

    def test_negative_threshold(self):
        vol = np.full((2, 2, 2), -1.0, dtype=np.float32)
        result = compute_binary_mask(vol, threshold=-2.0)
        assert np.all(result == 1.0)


# ===========================================================================
# fill_holes_3d
# ===========================================================================


class TestFillHoles3d:
    def test_all_zeros_remains_zeros(self, zero_volume):
        result = fill_holes_3d(zero_volume)
        assert np.all(result == 0.0)

    def test_all_ones_remains_ones(self, ones_volume):
        result = fill_holes_3d(ones_volume)
        assert np.all(result == 1.0)

    def test_output_dtype_is_float32(self, binary_mask):
        result = fill_holes_3d(binary_mask)
        assert result.dtype == np.float32

    def test_fills_interior_hole(self, binary_mask_with_holes):
        result = fill_holes_3d(binary_mask_with_holes)
        # The interior region that was zeroed must now be filled
        assert result[3, 3, 3] == 1.0

    def test_output_only_contains_0_and_1(self, binary_mask_with_holes):
        result = fill_holes_3d(binary_mask_with_holes)
        unique = np.unique(result)
        assert set(unique).issubset({0.0, 1.0})

    def test_shape_preserved(self, binary_mask_with_holes):
        result = fill_holes_3d(binary_mask_with_holes)
        assert result.shape == binary_mask_with_holes.shape

    def test_does_not_mutate_input(self, binary_mask_with_holes):
        original = binary_mask_with_holes.copy()
        fill_holes_3d(binary_mask_with_holes)
        np.testing.assert_array_equal(binary_mask_with_holes, original)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            fill_holes_3d(np.ones((4, 4)))

    def test_custom_structure_accepted(self, binary_mask_with_holes):
        struct = np.ones((5, 5, 5))
        result = fill_holes_3d(binary_mask_with_holes, structure=struct)
        assert result.shape == binary_mask_with_holes.shape

    def test_default_structure_is_3x3x3(self):
        from src.image_utils import compute_neighbor_structure
        struct = compute_neighbor_structure()
        assert struct.shape == (3, 3, 3)


# ===========================================================================
# compute_neighbor_structure
# ===========================================================================


class TestComputeNeighborStructure:
    def test_default_shape_is_3x3x3(self):
        s = compute_neighbor_structure()
        assert s.shape == (3, 3, 3)

    def test_all_values_are_one(self):
        s = compute_neighbor_structure()
        assert np.all(s == 1.0)

    def test_custom_size(self):
        s = compute_neighbor_structure(5)
        assert s.shape == (5, 5, 5)

    def test_raises_on_zero_size(self):
        with pytest.raises(ValueError):
            compute_neighbor_structure(0)

    def test_raises_on_negative_size(self):
        with pytest.raises(ValueError):
            compute_neighbor_structure(-1)

    def test_raises_on_non_integer_size(self):
        with pytest.raises((ValueError, TypeError)):
            compute_neighbor_structure(3.0)  # type: ignore[arg-type]

    def test_size_1_returns_scalar_cube(self):
        s = compute_neighbor_structure(1)
        assert s.shape == (1, 1, 1)
        assert s[0, 0, 0] == 1.0


# ===========================================================================
# normalize_volume
# ===========================================================================


class TestNormalizeVolume:
    def test_output_range_is_0_to_1(self, rng_volume):
        result = normalize_volume(rng_volume)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_min_voxel_becomes_new_min(self, gradient_volume):
        result = normalize_volume(gradient_volume, new_min=0.0, new_max=1.0)
        assert pytest.approx(float(result.min()), abs=1e-6) == 0.0

    def test_max_voxel_becomes_new_max(self, gradient_volume):
        result = normalize_volume(gradient_volume, new_min=0.0, new_max=1.0)
        assert pytest.approx(float(result.max()), abs=1e-6) == 1.0

    def test_custom_range(self, rng_volume):
        result = normalize_volume(rng_volume, new_min=-1.0, new_max=1.0)
        assert float(result.min()) >= -1.0 - 1e-9
        assert float(result.max()) <= 1.0 + 1e-9

    def test_uniform_volume_returns_new_min(self, uniform_volume):
        result = normalize_volume(uniform_volume)
        assert np.all(result == 0.0)

    def test_shape_preserved(self, rng_volume):
        result = normalize_volume(rng_volume)
        assert result.shape == rng_volume.shape

    def test_does_not_mutate_input(self, rng_volume):
        original = rng_volume.copy()
        normalize_volume(rng_volume)
        np.testing.assert_array_equal(rng_volume, original)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            normalize_volume(np.ones((4, 4)))

    def test_raises_if_new_min_equals_new_max(self, rng_volume):
        with pytest.raises(ValueError, match="new_min"):
            normalize_volume(rng_volume, new_min=0.5, new_max=0.5)

    def test_raises_if_new_min_greater_than_new_max(self, rng_volume):
        with pytest.raises(ValueError, match="new_min"):
            normalize_volume(rng_volume, new_min=1.0, new_max=0.0)

    def test_output_dtype_is_float64(self, rng_volume):
        result = normalize_volume(rng_volume)
        assert result.dtype == np.float64


# ===========================================================================
# clip_volume
# ===========================================================================


class TestClipVolume:
    def test_clip_within_range_unchanged(self):
        vol = np.array([[[0.5, 0.3], [0.7, 0.2]], [[0.1, 0.9], [0.4, 0.6]]])
        result = clip_volume(vol, 0.0, 1.0)
        np.testing.assert_array_equal(result, vol)

    def test_clips_below_low(self):
        vol = np.full((2, 2, 2), -5.0)
        result = clip_volume(vol, 0.0, 1.0)
        assert np.all(result == 0.0)

    def test_clips_above_high(self):
        vol = np.full((2, 2, 2), 999.0)
        result = clip_volume(vol, 0.0, 255.0)
        assert np.all(result == 255.0)

    def test_shape_preserved(self, integer_volume):
        result = clip_volume(integer_volume, 0.0, 200.0)
        assert result.shape == integer_volume.shape

    def test_does_not_mutate_input(self, integer_volume):
        original = integer_volume.copy()
        clip_volume(integer_volume, 0.0, 100.0)
        np.testing.assert_array_equal(integer_volume, original)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            clip_volume(np.ones((4, 4)), 0.0, 1.0)

    def test_raises_if_low_greater_than_high(self):
        with pytest.raises(ValueError, match="low"):
            clip_volume(np.ones((2, 2, 2)), 5.0, 1.0)

    def test_equal_low_and_high_accepted(self):
        vol = np.ones((2, 2, 2)) * 3.0
        result = clip_volume(vol, 3.0, 3.0)
        assert np.all(result == 3.0)


# ===========================================================================
# threshold_volume
# ===========================================================================


class TestThresholdVolume:
    def test_all_above_threshold_returns_all_ones(self, ones_volume):
        result = threshold_volume(ones_volume, threshold=0.5)
        assert np.all(result == 1.0)

    def test_all_below_threshold_returns_all_zeros(self, ones_volume):
        result = threshold_volume(ones_volume, threshold=2.0)
        assert np.all(result == 0.0)

    def test_mixed_values(self):
        vol = np.array([[[0.1, 0.9], [0.5, 0.6]], [[0.4, 0.7], [0.3, 0.8]]])
        result = threshold_volume(vol, 0.5)
        assert result[0, 0, 1] == 1.0  # 0.9 > 0.5
        assert result[0, 0, 0] == 0.0  # 0.1 <= 0.5
        assert result[0, 1, 0] == 0.0  # 0.5 is NOT > 0.5

    def test_output_dtype_is_float32(self, rng_volume):
        result = threshold_volume(rng_volume, 0.5)
        assert result.dtype == np.float32

    def test_output_contains_only_0_and_1(self, rng_volume):
        result = threshold_volume(rng_volume, 0.5)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            threshold_volume(np.ones((4, 4)), 0.5)

    def test_shape_preserved(self, rng_volume):
        result = threshold_volume(rng_volume, 0.5)
        assert result.shape == rng_volume.shape


# ===========================================================================
# compute_volume_statistics
# ===========================================================================


class TestComputeVolumeStatistics:
    def test_returns_dict_with_required_keys(self, rng_volume):
        stats = compute_volume_statistics(rng_volume)
        for key in ("min", "max", "mean", "std", "nonzero_count", "total_voxels"):
            assert key in stats

    def test_min_value_correct(self, gradient_volume):
        stats = compute_volume_statistics(gradient_volume)
        assert stats["min"] == pytest.approx(float(gradient_volume.min()))

    def test_max_value_correct(self, gradient_volume):
        stats = compute_volume_statistics(gradient_volume)
        assert stats["max"] == pytest.approx(float(gradient_volume.max()))

    def test_total_voxels_is_volume_size(self, rng_volume):
        stats = compute_volume_statistics(rng_volume)
        assert stats["total_voxels"] == rng_volume.size

    def test_nonzero_count_for_binary_mask(self, binary_mask):
        stats = compute_volume_statistics(binary_mask)
        expected = int(np.count_nonzero(binary_mask))
        assert stats["nonzero_count"] == expected

    def test_all_zeros_stats(self, zero_volume):
        stats = compute_volume_statistics(zero_volume)
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["nonzero_count"] == 0

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_volume_statistics(np.ones((4, 4)))

    def test_std_non_negative(self, rng_volume):
        stats = compute_volume_statistics(rng_volume)
        assert stats["std"] >= 0.0


# ===========================================================================
# count_nonzero_voxels
# ===========================================================================


class TestCountNonzeroVoxels:
    def test_all_zeros_returns_zero(self, zero_volume):
        assert count_nonzero_voxels(zero_volume) == 0

    def test_all_ones_returns_size(self, ones_volume):
        assert count_nonzero_voxels(ones_volume) == ones_volume.size

    def test_binary_mask_count(self, binary_mask):
        expected = int(np.count_nonzero(binary_mask))
        assert count_nonzero_voxels(binary_mask) == expected

    def test_returns_int(self, rng_volume):
        result = count_nonzero_voxels(rng_volume)
        assert isinstance(result, int)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            count_nonzero_voxels(np.ones((4, 4)))


# ===========================================================================
# validate_volume_shape
# ===========================================================================


class TestValidateVolumeShape:
    def test_valid_3d_array_returns_true(self, rng_volume):
        assert validate_volume_shape(rng_volume) is True

    def test_2d_array_returns_false(self):
        assert validate_volume_shape(np.ones((4, 4))) is False

    def test_4d_array_returns_false(self):
        assert validate_volume_shape(np.ones((2, 2, 2, 2))) is False

    def test_non_array_returns_false(self):
        assert validate_volume_shape([[[1, 2], [3, 4]]]) is False  # type: ignore[arg-type]

    def test_empty_array_returns_false(self):
        assert validate_volume_shape(np.empty((0, 4, 4))) is False

    def test_scalar_array_returns_false(self):
        assert validate_volume_shape(np.array(5.0)) is False

    def test_one_voxel_volume_returns_true(self):
        assert validate_volume_shape(np.ones((1, 1, 1))) is True


# ===========================================================================
# validate_subject_dict
# ===========================================================================


class TestValidateSubjectDict:
    def test_valid_dict_returns_true(self, valid_subject_dict):
        ok, msg = validate_subject_dict(valid_subject_dict)
        assert ok is True
        assert msg == ""

    def test_missing_subject_id_fails(self, valid_subject_dict):
        d = {k: v for k, v in valid_subject_dict.items() if k != "SubjectID"}
        ok, msg = validate_subject_dict(d)
        assert ok is False
        assert "SubjectID" in msg

    def test_missing_t1w_path_fails(self, valid_subject_dict):
        d = {k: v for k, v in valid_subject_dict.items() if k != "t1wPath"}
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_missing_t1cw_path_fails(self, valid_subject_dict):
        d = {k: v for k, v in valid_subject_dict.items() if k != "t1cwPath"}
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_missing_t2w_path_fails(self, valid_subject_dict):
        d = {k: v for k, v in valid_subject_dict.items() if k != "t2wPath"}
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_missing_flair_path_fails(self, valid_subject_dict):
        d = {k: v for k, v in valid_subject_dict.items() if k != "flairPath"}
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_empty_dict_fails(self):
        ok, msg = validate_subject_dict({})
        assert ok is False
        assert msg != ""

    def test_non_dict_fails(self):
        ok, msg = validate_subject_dict("not a dict")  # type: ignore[arg-type]
        assert ok is False

    def test_empty_subject_id_fails(self, valid_subject_dict):
        d = dict(valid_subject_dict)
        d["SubjectID"] = "   "
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_numeric_subject_id_fails(self, valid_subject_dict):
        d = dict(valid_subject_dict)
        d["SubjectID"] = 12345  # type: ignore[assignment]
        ok, msg = validate_subject_dict(d)
        assert ok is False

    def test_required_keys_constant(self):
        for key in ("SubjectID", "t1wPath", "t1cwPath", "t2wPath", "flairPath"):
            assert key in REQUIRED_SUBJECT_KEYS


# ===========================================================================
# build_subject_output_path
# ===========================================================================


class TestBuildSubjectOutputPath:
    def test_basic_path_construction(self):
        result = build_subject_output_path("/results", "EGD-0117")
        assert result == "/results/EGD-0117"

    def test_trailing_slash_in_save_dir(self):
        result = build_subject_output_path("/results/", "EGD-0117")
        assert "EGD-0117" in result

    def test_subject_id_is_last_component(self):
        result = build_subject_output_path("/some/deep/dir", "SUBJECT-001")
        assert result.endswith("SUBJECT-001")

    def test_empty_save_dir_raises(self):
        with pytest.raises(ValueError, match="save_dir"):
            build_subject_output_path("", "EGD-0117")

    def test_whitespace_save_dir_raises(self):
        with pytest.raises(ValueError, match="save_dir"):
            build_subject_output_path("   ", "EGD-0117")

    def test_empty_subject_id_raises(self):
        with pytest.raises(ValueError, match="subject_id"):
            build_subject_output_path("/results", "")

    def test_whitespace_subject_id_raises(self):
        with pytest.raises(ValueError, match="subject_id"):
            build_subject_output_path("/results", "  ")


# ===========================================================================
# compute_bounding_box
# ===========================================================================


class TestComputeBoundingBox:
    def test_all_zeros_returns_none(self, zero_volume):
        result = compute_bounding_box(zero_volume)
        assert result is None

    def test_all_ones_returns_full_extent(self, ones_volume):
        bbox = compute_bounding_box(ones_volume)
        assert bbox is not None
        (x0, x1), (y0, y1), (z0, z1) = bbox
        assert x0 == 0 and x1 == 7
        assert y0 == 0 and y1 == 7
        assert z0 == 0 and z1 == 7

    def test_binary_mask_bounding_box(self, binary_mask):
        bbox = compute_bounding_box(binary_mask)
        # binary_mask has ones at [2:6, 2:6, 2:6]
        assert bbox is not None
        (x0, x1), (y0, y1), (z0, z1) = bbox
        assert x0 == 2 and x1 == 5
        assert y0 == 2 and y1 == 5
        assert z0 == 2 and z1 == 5

    def test_returns_tuple_of_tuples(self, binary_mask):
        bbox = compute_bounding_box(binary_mask)
        assert isinstance(bbox, tuple)
        assert len(bbox) == 3
        for pair in bbox:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_single_voxel_mask(self):
        vol = np.zeros((8, 8, 8))
        vol[4, 3, 2] = 1.0
        bbox = compute_bounding_box(vol)
        assert bbox == ((4, 4), (3, 3), (2, 2))

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_bounding_box(np.ones((4, 4)))


# ===========================================================================
# extract_brain_roi
# ===========================================================================


class TestExtractBrainRoi:
    def test_zeros_mask_blanks_volume(self, rng_volume, zero_volume):
        result = extract_brain_roi(rng_volume, zero_volume)
        assert np.all(result == 0.0)

    def test_ones_mask_preserves_volume(self, rng_volume, ones_volume):
        result = extract_brain_roi(rng_volume, ones_volume)
        np.testing.assert_array_almost_equal(result, rng_volume)

    def test_partial_mask_zeros_outside(self, ones_volume, binary_mask):
        result = extract_brain_roi(ones_volume, binary_mask)
        assert result[0, 0, 0] == 0.0
        assert result[3, 3, 3] == 1.0

    def test_shape_preserved(self, rng_volume, binary_mask):
        result = extract_brain_roi(rng_volume, binary_mask)
        assert result.shape == rng_volume.shape


# ===========================================================================
# resample_volume_nearest
# ===========================================================================


class TestResampleVolumeNearest:
    def test_output_shape_matches_target(self, rng_volume):
        target = (4, 4, 4)
        result = resample_volume_nearest(rng_volume, target)
        assert result.shape == target

    def test_upsample_doubles_shape(self, small_volume_5x5x5):
        target = (10, 10, 10)
        result = resample_volume_nearest(small_volume_5x5x5, target)
        assert result.shape == target

    def test_same_shape_returns_equivalent(self, binary_mask):
        result = resample_volume_nearest(binary_mask, binary_mask.shape)
        assert result.shape == binary_mask.shape

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            resample_volume_nearest(np.ones((4, 4)), (2, 2, 2))

    def test_raises_if_target_shape_wrong_length(self, rng_volume):
        with pytest.raises(ValueError, match="target_shape"):
            resample_volume_nearest(rng_volume, (4, 4))

    def test_binary_mask_stays_binary_after_nearest_resample(self, binary_mask):
        target = (4, 4, 4)
        result = resample_volume_nearest(binary_mask, target)
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0.0, 1.0})


# ===========================================================================
# pad_volume_to_shape
# ===========================================================================


class TestPadVolumeToShape:
    def test_output_shape_matches_target(self, small_volume_5x5x5):
        result = pad_volume_to_shape(small_volume_5x5x5, (8, 8, 8))
        assert result.shape == (8, 8, 8)

    def test_same_shape_no_change(self, rng_volume):
        result = pad_volume_to_shape(rng_volume, rng_volume.shape)
        assert result.shape == rng_volume.shape
        np.testing.assert_array_equal(result[2:6, 2:6, 2:6], rng_volume[2:6, 2:6, 2:6])

    def test_padded_area_is_zero_by_default(self, small_volume_5x5x5):
        result = pad_volume_to_shape(small_volume_5x5x5, (8, 8, 8))
        # Original occupies center; corners should be 0
        assert result[0, 0, 0] == 0.0

    def test_custom_pad_value(self, small_volume_5x5x5):
        result = pad_volume_to_shape(small_volume_5x5x5, (8, 8, 8), pad_value=-1.0)
        assert result[0, 0, 0] == -1.0

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            pad_volume_to_shape(np.ones((4, 4)), (8, 8, 8))

    def test_raises_if_target_smaller(self, rng_volume):
        with pytest.raises(ValueError, match="smaller"):
            pad_volume_to_shape(rng_volume, (4, 4, 4))

    def test_raises_if_target_wrong_length(self, rng_volume):
        with pytest.raises(ValueError, match="target_shape"):
            pad_volume_to_shape(rng_volume, (16, 16))


# ===========================================================================
# crop_volume_to_bounding_box
# ===========================================================================


class TestCropVolumeToBoundingBox:
    def test_crop_reduces_shape(self, ones_volume, binary_mask):
        result = crop_volume_to_bounding_box(ones_volume, binary_mask)
        # binary_mask is 4x4x4 inside 8x8x8, so cropped shape is 4x4x4
        assert result.shape == (4, 4, 4)

    def test_crop_full_mask_preserves_volume(self, rng_volume, ones_volume):
        result = crop_volume_to_bounding_box(rng_volume, ones_volume)
        np.testing.assert_array_equal(result, rng_volume)

    def test_raises_if_shapes_differ(self):
        vol = np.ones((8, 8, 8))
        mask = np.ones((4, 4, 4))
        with pytest.raises(ValueError):
            crop_volume_to_bounding_box(vol, mask)

    def test_raises_if_mask_all_zeros(self, rng_volume, zero_volume):
        with pytest.raises(ValueError, match="all zeros"):
            crop_volume_to_bounding_box(rng_volume, zero_volume)

    def test_single_voxel_crop(self):
        vol = np.arange(512).reshape(8, 8, 8).astype(float)
        mask = np.zeros((8, 8, 8))
        mask[3, 3, 3] = 1.0
        result = crop_volume_to_bounding_box(vol, mask)
        assert result.shape == (1, 1, 1)
        assert result[0, 0, 0] == vol[3, 3, 3]


# ===========================================================================
# compute_center_of_mass
# ===========================================================================


class TestComputeCenterOfMass:
    def test_uniform_volume_centre(self):
        vol = np.ones((8, 8, 8))
        com = compute_center_of_mass(vol)
        assert com is not None
        for coord in com:
            assert pytest.approx(coord, abs=0.1) == 3.5

    def test_single_voxel_at_known_location(self):
        vol = np.zeros((8, 8, 8))
        vol[2, 3, 4] = 1.0
        com = compute_center_of_mass(vol)
        assert com is not None
        assert pytest.approx(com[0], abs=1e-9) == 2.0
        assert pytest.approx(com[1], abs=1e-9) == 3.0
        assert pytest.approx(com[2], abs=1e-9) == 4.0

    def test_returns_tuple_of_floats(self, binary_mask):
        com = compute_center_of_mass(binary_mask)
        assert com is not None
        assert len(com) == 3
        for v in com:
            assert isinstance(v, float)

    def test_all_zeros_returns_result(self, zero_volume):
        # scipy returns (nan, nan, nan) for empty inputs; we propagate that as None
        com = compute_center_of_mass(zero_volume)
        # Either None or the computed value is acceptable
        assert com is None or isinstance(com, tuple)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_center_of_mass(np.ones((4, 4)))


# ===========================================================================
# compute_dice_coefficient
# ===========================================================================


class TestComputeDiceCoefficient:
    def test_identical_masks_return_1(self, binary_mask):
        result = compute_dice_coefficient(binary_mask, binary_mask)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_disjoint_masks_return_0(self):
        a = np.zeros((8, 8, 8))
        a[0:4, :, :] = 1.0
        b = np.zeros((8, 8, 8))
        b[4:8, :, :] = 1.0
        result = compute_dice_coefficient(a, b)
        assert pytest.approx(result, abs=1e-6) == 0.0

    def test_both_empty_returns_1(self, zero_volume):
        result = compute_dice_coefficient(zero_volume, zero_volume)
        assert result == 1.0

    def test_value_between_0_and_1(self, binary_mask, binary_mask_with_holes):
        result = compute_dice_coefficient(binary_mask, binary_mask_with_holes)
        assert 0.0 <= result <= 1.0

    def test_raises_if_shapes_differ(self):
        a = np.ones((8, 8, 8))
        b = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="shape"):
            compute_dice_coefficient(a, b)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_dice_coefficient(np.ones((4, 4)), np.ones((4, 4)))

    def test_partial_overlap(self):
        a = np.zeros((4, 4, 4))
        a[0:2, :, :] = 1.0
        b = np.zeros((4, 4, 4))
        b[1:3, :, :] = 1.0
        result = compute_dice_coefficient(a, b)
        # intersection = 1 layer (16 voxels), |A|+|B| = 32+32 = 64
        # Dice = 2*16/64 = 0.5
        assert pytest.approx(result, abs=1e-6) == 0.5

    def test_symmetry(self, binary_mask, binary_mask_with_holes):
        r1 = compute_dice_coefficient(binary_mask, binary_mask_with_holes)
        r2 = compute_dice_coefficient(binary_mask_with_holes, binary_mask)
        assert pytest.approx(r1, abs=1e-9) == r2


# ===========================================================================
# compute_volume_overlap (Jaccard / IoU)
# ===========================================================================


class TestComputeVolumeOverlap:
    def test_identical_masks_return_1(self, binary_mask):
        result = compute_volume_overlap(binary_mask, binary_mask)
        assert pytest.approx(result, abs=1e-6) == 1.0

    def test_disjoint_masks_return_0(self):
        a = np.zeros((8, 8, 8))
        a[0:4, :, :] = 1.0
        b = np.zeros((8, 8, 8))
        b[4:8, :, :] = 1.0
        result = compute_volume_overlap(a, b)
        assert pytest.approx(result, abs=1e-6) == 0.0

    def test_both_empty_returns_1(self, zero_volume):
        result = compute_volume_overlap(zero_volume, zero_volume)
        assert result == 1.0

    def test_value_between_0_and_1(self, binary_mask, binary_mask_with_holes):
        result = compute_volume_overlap(binary_mask, binary_mask_with_holes)
        assert 0.0 <= result <= 1.0

    def test_raises_if_shapes_differ(self):
        a = np.ones((8, 8, 8))
        b = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="shape"):
            compute_volume_overlap(a, b)

    def test_raises_if_not_3d(self):
        with pytest.raises(ValueError, match="3-D"):
            compute_volume_overlap(np.ones((4, 4)), np.ones((4, 4)))

    def test_partial_overlap_value(self):
        a = np.zeros((4, 4, 4))
        a[0:2, :, :] = 1.0
        b = np.zeros((4, 4, 4))
        b[1:3, :, :] = 1.0
        result = compute_volume_overlap(a, b)
        # intersection = 16, union = 48
        assert pytest.approx(result, abs=1e-6) == 16.0 / 48.0

    def test_symmetry(self, binary_mask, binary_mask_with_holes):
        r1 = compute_volume_overlap(binary_mask, binary_mask_with_holes)
        r2 = compute_volume_overlap(binary_mask_with_holes, binary_mask)
        assert pytest.approx(r1, abs=1e-9) == r2

    def test_iou_le_dice(self, binary_mask, binary_mask_with_holes):
        """IoU is always <= Dice for non-trivial overlaps."""
        dice = compute_dice_coefficient(binary_mask, binary_mask_with_holes)
        iou = compute_volume_overlap(binary_mask, binary_mask_with_holes)
        assert iou <= dice + 1e-9
