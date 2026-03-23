"""
Notebook validity tests.

Each notebook is parsed as raw JSON to verify:
  - Valid JSON syntax
  - Correct nbformat version
  - Presence of required metadata (kernelspec, language_info)
  - Expected cell counts and cell types
  - Key function definitions in code cells
  - Required imports appear somewhere in the notebook

No notebook kernel is launched; tests are purely structural.
"""

import json
import os

import pytest

from conftest import NOTEBOOK_FILENAMES, NOTEBOOKS_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_notebook(filename: str) -> dict:
    path = os.path.join(NOTEBOOKS_DIR, filename)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_code_sources(notebook: dict) -> list[str]:
    """Return the concatenated source of every code cell."""
    return [
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    ]


def get_all_source(notebook: dict) -> str:
    """Concatenate all cell sources into one string for keyword searches."""
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


# ---------------------------------------------------------------------------
# Parametrized: all notebooks must be valid JSON with correct structure
# ---------------------------------------------------------------------------


class TestAllNotebooksValidJson:
    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_is_valid_json(self, filename):
        path = os.path.join(NOTEBOOKS_DIR, filename)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        assert isinstance(data, dict), f"{filename} root must be a JSON object"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_has_cells_key(self, filename):
        nb = load_notebook(filename)
        assert "cells" in nb, f"{filename} must have a 'cells' key"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_cells_is_list(self, filename):
        nb = load_notebook(filename)
        assert isinstance(nb["cells"], list), f"{filename} 'cells' must be a list"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_has_nbformat_key(self, filename):
        nb = load_notebook(filename)
        assert "nbformat" in nb, f"{filename} must declare 'nbformat'"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_nbformat_is_4(self, filename):
        nb = load_notebook(filename)
        assert nb["nbformat"] == 4, f"{filename} nbformat must be 4"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_has_metadata(self, filename):
        nb = load_notebook(filename)
        assert "metadata" in nb, f"{filename} must have 'metadata'"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_metadata_has_kernelspec(self, filename):
        nb = load_notebook(filename)
        assert "kernelspec" in nb.get("metadata", {}), (
            f"{filename} metadata must have 'kernelspec'"
        )

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_kernelspec_language_is_python(self, filename):
        nb = load_notebook(filename)
        lang = nb["metadata"]["kernelspec"].get("language", "")
        assert lang == "python", f"{filename} kernelspec language must be 'python'"

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_metadata_has_language_info(self, filename):
        nb = load_notebook(filename)
        assert "language_info" in nb.get("metadata", {}), (
            f"{filename} metadata must have 'language_info'"
        )

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_language_info_name_is_python(self, filename):
        nb = load_notebook(filename)
        lang_name = nb["metadata"]["language_info"].get("name", "")
        assert lang_name == "python", (
            f"{filename} language_info.name must be 'python'"
        )

    @pytest.mark.parametrize("filename", NOTEBOOK_FILENAMES)
    def test_cells_have_required_fields(self, filename):
        nb = load_notebook(filename)
        for i, cell in enumerate(nb["cells"]):
            assert "cell_type" in cell, (
                f"{filename} cell {i} missing 'cell_type'"
            )
            assert "source" in cell, (
                f"{filename} cell {i} missing 'source'"
            )


# ---------------------------------------------------------------------------
# 3D-image-registration.ipynb  (6 code cells, 6 markdown cells)
# ---------------------------------------------------------------------------


class TestRegistrationNotebook:
    FILENAME = "3D-image-registration.ipynb"

    def test_total_cell_count(self):
        nb = load_notebook(self.FILENAME)
        assert len(nb["cells"]) == 12

    def test_code_cell_count(self):
        nb = load_notebook(self.FILENAME)
        code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code) == 6

    def test_markdown_cell_count(self):
        nb = load_notebook(self.FILENAME)
        md = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
        assert len(md) == 6

    def test_defines_brats_registration_function(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "def brats_ants_mni152betafter_registration" in src

    def test_imports_numpy(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "import numpy" in src

    def test_imports_scipy_ndimage(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "ndimage" in src

    def test_uses_n4_bias_correction(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "n4_bias_field_correction" in src

    def test_uses_dense_rigid_transform(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "DenseRigid" in src

    def test_references_hd_bet(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "hd-bet" in src

    def test_references_atlas_path(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "AtlasPath" in src


# ---------------------------------------------------------------------------
# BraTS-sri24-AtlasProcessBrainExtract.ipynb
# ---------------------------------------------------------------------------


class TestAtlasPreprocessingNotebook:
    FILENAME = "BraTS-sri24-AtlasProcessBrainExtract.ipynb"

    def test_total_cell_count(self):
        nb = load_notebook(self.FILENAME)
        assert len(nb["cells"]) == 28

    def test_code_cell_count(self):
        nb = load_notebook(self.FILENAME)
        code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code) == 17

    def test_markdown_cell_count(self):
        nb = load_notebook(self.FILENAME)
        md = [c for c in nb["cells"] if c.get("cell_type") == "markdown"]
        assert len(md) == 11

    def test_defines_get_image_basic_info(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "def get_ImageBasicInfo" in src

    def test_references_sri24_atlas(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "sri24" in src

    def test_references_brats_origin(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "brats_origin" in src

    def test_references_hd_bet(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "hd-bet" in src


# ---------------------------------------------------------------------------
# antsPyRegistrationBasic.ipynb
# ---------------------------------------------------------------------------


class TestAntsPyTutorialNotebook:
    FILENAME = "antsPyRegistrationBasic.ipynb"

    def test_total_cell_count(self):
        nb = load_notebook(self.FILENAME)
        assert len(nb["cells"]) == 16

    def test_code_cell_count(self):
        nb = load_notebook(self.FILENAME)
        code = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        assert len(code) == 10

    def test_uses_registration_call(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "ants.registration" in src

    def test_uses_apply_transforms(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "ants.apply_transforms" in src

    def test_uses_syn_transform(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "SyN" in src

    def test_references_r16_image(self):
        nb = load_notebook(self.FILENAME)
        src = get_all_source(nb)
        assert "r16" in src
