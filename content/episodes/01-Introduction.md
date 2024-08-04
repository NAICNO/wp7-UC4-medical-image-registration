# Introduction

This use case demonstrates 3D medical image registration and segmentation
using synthetic brain-like volumes. The pipeline covers:

1. **Volume generation** — synthetic T1/T2 weighted MRI-like 3D arrays
2. **Preprocessing** — intensity normalization, Gaussian smoothing
3. **Registration** — rigid alignment of misaligned volumes
4. **Segmentation** — intensity thresholding and region labeling

All operations use NumPy and SciPy only — no heavy medical imaging
libraries (ANTsPy, nibabel) are required for the demonstrator.
