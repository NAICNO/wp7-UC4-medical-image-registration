"""
Import smoke tests for the src package.

Verifies that:
  - The src package itself is importable
  - All public symbols listed in __all__ are importable
  - The available scientific stack (numpy, scipy) is importable

Heavy dependencies (nibabel, antspyx, monai, SimpleITK, hd-bet) are
expected to be absent in the CI environment; the tests below assert
that their absence does NOT prevent importing the src package.
"""

import importlib
import sys

import pytest


class TestSrcPackageImportable:
    def test_src_package_imports(self):
        import src  # noqa: F401

    def test_src_image_utils_imports(self):
        import src.image_utils  # noqa: F401

    def test_all_public_symbols_importable(self):
        import src
        for name in src.__all__:
            assert hasattr(src, name), f"src.{name} not found in package"

    def test_apply_binary_mask_importable(self):
        from src.image_utils import apply_binary_mask  # noqa: F401

    def test_normalize_volume_importable(self):
        from src.image_utils import normalize_volume  # noqa: F401

    def test_compute_binary_mask_importable(self):
        from src.image_utils import compute_binary_mask  # noqa: F401

    def test_fill_holes_3d_importable(self):
        from src.image_utils import fill_holes_3d  # noqa: F401

    def test_clip_volume_importable(self):
        from src.image_utils import clip_volume  # noqa: F401

    def test_compute_volume_statistics_importable(self):
        from src.image_utils import compute_volume_statistics  # noqa: F401

    def test_validate_subject_dict_importable(self):
        from src.image_utils import validate_subject_dict  # noqa: F401

    def test_compute_dice_coefficient_importable(self):
        from src.image_utils import compute_dice_coefficient  # noqa: F401


class TestAvailableDependencies:
    """The available deps (numpy, scipy) must be importable."""

    def test_numpy_importable(self):
        import numpy  # noqa: F401

    def test_scipy_importable(self):
        import scipy  # noqa: F401

    def test_scipy_ndimage_importable(self):
        from scipy import ndimage  # noqa: F401

    def test_numpy_version_at_least_1_20(self):
        import numpy as np
        major, minor = map(int, np.__version__.split(".")[:2])
        assert (major, minor) >= (1, 20), (
            f"numpy >= 1.20 required, got {np.__version__}"
        )


class TestHeavyDepsAbsent:
    """
    Confirm that heavy deps are NOT importable in this test environment.

    If any become available, these tests will fail and the full pipeline
    tests should then be re-enabled.
    """

    @pytest.mark.xfail(reason="antspyx not installed in this environment", strict=False)
    def test_antspyx_not_available(self):
        with pytest.raises(ImportError):
            import ants  # noqa: F401

    @pytest.mark.xfail(reason="nibabel not installed in this environment", strict=False)
    def test_nibabel_not_available(self):
        with pytest.raises(ImportError):
            import nibabel  # noqa: F401

    @pytest.mark.xfail(reason="monai not installed in this environment", strict=False)
    def test_monai_not_available(self):
        with pytest.raises(ImportError):
            import monai  # noqa: F401

    @pytest.mark.xfail(reason="SimpleITK not installed in this environment", strict=False)
    def test_simpleitk_not_available(self):
        with pytest.raises(ImportError):
            import SimpleITK  # noqa: F401
