"""
Project structure validation tests.

Verifies that all expected files and directories exist and have the
correct names/extensions.  These tests can catch accidental renames,
missing assets, or broken repo clones without executing any model code.

TDD cycle:
  RED  - define expectations of the repo layout
  GREEN - structure already exists (tests pass from the start)
  REFACTOR - add more assertions as the repo grows
"""

import json
import os

import pytest

from conftest import (
    ASSETS_DIR,
    DATA_DIR,
    NOTEBOOK_FILENAMES,
    NOTEBOOKS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SRC_DIR,
)


# ---------------------------------------------------------------------------
# Root-level files
# ---------------------------------------------------------------------------


class TestRootLevelFiles:
    def test_readme_exists(self):
        assert os.path.isfile(f"{PROJECT_ROOT}/README.md"), "README.md must exist"

    def test_environment_yml_exists(self):
        assert os.path.isfile(
            f"{PROJECT_ROOT}/environment.yml"
        ), "environment.yml must exist"

    def test_environment_yml_is_not_empty(self):
        path = f"{PROJECT_ROOT}/environment.yml"
        assert os.path.getsize(path) > 0, "environment.yml must not be empty"


# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------


class TestTopLevelDirectories:
    def test_notebooks_dir_exists(self):
        assert os.path.isdir(NOTEBOOKS_DIR), "notebooks/ directory must exist"

    def test_data_dir_exists(self):
        assert os.path.isdir(DATA_DIR), "data/ directory must exist"

    def test_assets_dir_exists(self):
        assert os.path.isdir(ASSETS_DIR), "assets/ directory must exist"

    def test_results_dir_exists(self):
        assert os.path.isdir(RESULTS_DIR), "results/ directory must exist"

    def test_src_dir_exists(self):
        assert os.path.isdir(SRC_DIR), "src/ directory must exist"

    def test_tests_dir_exists(self):
        tests_dir = f"{PROJECT_ROOT}/tests"
        assert os.path.isdir(tests_dir), "tests/ directory must exist"


# ---------------------------------------------------------------------------
# Notebook files
# ---------------------------------------------------------------------------


class TestNotebookFiles:
    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_notebook_file_exists(self, filename):
        path = os.path.join(NOTEBOOKS_DIR, filename)
        assert os.path.isfile(path), f"{filename} must exist in notebooks/"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_notebook_has_ipynb_extension(self, filename):
        assert filename.endswith(".ipynb"), f"{filename} must have .ipynb extension"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_notebook_is_not_empty(self, filename):
        path = os.path.join(NOTEBOOKS_DIR, filename)
        assert os.path.getsize(path) > 0, f"{filename} must not be empty"

    def test_three_notebooks_present(self):
        nb_files = [
            f
            for f in os.listdir(NOTEBOOKS_DIR)
            if f.endswith(".ipynb")
        ]
        assert len(nb_files) == 3, "There must be exactly 3 notebooks"


# ---------------------------------------------------------------------------
# src package
# ---------------------------------------------------------------------------


class TestSrcPackage:
    def test_src_init_exists(self):
        assert os.path.isfile(
            f"{SRC_DIR}/__init__.py"
        ), "src/__init__.py must exist"

    def test_src_image_utils_exists(self):
        assert os.path.isfile(
            f"{SRC_DIR}/image_utils.py"
        ), "src/image_utils.py must exist"

    def test_src_image_utils_is_not_empty(self):
        path = f"{SRC_DIR}/image_utils.py"
        assert os.path.getsize(path) > 0, "src/image_utils.py must not be empty"


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------


class TestAssets:
    def test_registration_diagram_exists(self):
        assert os.path.isfile(
            f"{ASSETS_DIR}/registration-explain.png"
        ), "assets/registration-explain.png must exist"

    def test_registration_diagram_is_non_empty_png(self):
        path = f"{ASSETS_DIR}/registration-explain.png"
        assert os.path.getsize(path) > 0, "registration-explain.png must not be empty"


# ---------------------------------------------------------------------------
# environment.yml content
# ---------------------------------------------------------------------------


class TestEnvironmentYml:
    def _load_yaml_lines(self):
        path = f"{PROJECT_ROOT}/environment.yml"
        with open(path) as fh:
            return fh.read()

    def test_env_name_is_present(self):
        content = self._load_yaml_lines()
        assert "name:" in content, "environment.yml must declare 'name:'"

    def test_env_name_value(self):
        content = self._load_yaml_lines()
        assert "3d-image-registration-segmentation" in content, (
            "environment name must be '3d-image-registration-segmentation'"
        )

    def test_python_version_specified(self):
        content = self._load_yaml_lines()
        assert "python=3.11" in content, "environment.yml must pin Python 3.11"

    def test_pip_section_present(self):
        content = self._load_yaml_lines()
        assert "pip:" in content, "environment.yml must have a pip section"

    def test_antspyx_listed(self):
        content = self._load_yaml_lines()
        assert "antspyx" in content, "antspyx must be listed in environment.yml"

    def test_nibabel_listed(self):
        content = self._load_yaml_lines()
        assert "nibabel" in content, "nibabel must be listed in environment.yml"
