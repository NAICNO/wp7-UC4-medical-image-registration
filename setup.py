"""
Minimal setup.py so that `src` is importable when running pytest from
the project root without installing the package.

Usage:
    pip install -e .   # editable install
    # or just run pytest from the project root; conftest.py/pytest.ini
    # ensure sys.path includes the project root.
"""

from setuptools import find_packages, setup

setup(
    name="3d-image-registration-segmentation",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=7",
            "pytest-cov",
        ]
    },
)
