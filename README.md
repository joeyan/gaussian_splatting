# 3D Gaussian Splatting
A "from scratch" re-implementation of [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl and Kopanas et al.


## Installation
This package requires CUDA and PyTorch.

1. CUDA install instructions [here](https://developer.nvidia.com/cuda-downloads)

2. PyTorch install instructions [here](https://pytorch.org/get-started/locally/)

3. Install the PyTorch CUDA extension
```
pip install -e ./
```
Note:
- Windows systems may need modify compilation flags in `setup.py`
- This step may be sensitive to the version of `pip`. This step failed after upgrading from `23.0.1` to `23.3.2`


## Verifying Install

1. Download the[Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset and unzip

```
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip && unzip 360_v2.zip
```

2. Update the `DATASET_PATH` in `splat_py/constants.py`

3. Run `colmap_splat.py`


To run all unit tests:

```
python -m unittest discover test
```
