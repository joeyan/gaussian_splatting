# 3D Gaussian Splatting
A "from scratch" re-implementation of [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl and Kopanas et al.

This repository implements the forward and backwards passes using a PyTorch CUDA extension based on the algorithms descriped in the paper. Some details of the splatting and adaptive control algorithm are not explicitly described in the paper and there may be differences between this repo and the official implementation.

The forward and backward pass algorithms are detailed in MATH.md


## Performance

Evaluations done with the Mip-NeRF 360 dataset at ~1 megapixel resoloution. This corresponds to the 2x downsampled indoor scenes and 4x downsampled outdoor scenes. Every 8th image was used for the test split.


Here are some comparisons with the with the official implementation. 


| Method       | Dataset     | PSNR | SSIM | N Gaussians | Train Duration*  |
|--------------|-------------|------|------|-------------|------------------|
| Official-30k | Garden 1/4x | 27.41| 0.87 |             | ~35-45min (A6000)|
| Ours-30k     | Garden 1/4x | 26.73| 0.83 | 2.02M       | ~31min (RTX4090) |
| Official-30k | Counter 1/2x| 28.70| 0.90 |             |                  |
| Ours-30k     | Counter 1/2x| 28.28| 0.90 | 1.47M       | ~35min (RTX4090) |
| Official-30k | Bonsai  1/2x| 31.98| 0.94 |             |                  |
| Ours-30k     | Bonsai  1/2x| 31.65| 0.94 | 2.01M       | ~45min (RTX4090) |
| Official-7k  | Bonsai 1/2x | 28.85| 0.91 |             |                  |
| Ours-7k      | Bonsai 1/2x | 29.84| 0.93 | 1.65M       | ~7min  (RTX4090) |
| Official-7k  | Garden 1/4x | 26.24| 0.83 |             |                  |
| Ours-7k      | Garden 1/4x | 25.47| 0.78 | 1.13M       | ~5min  (RTX4090) |


*The training time is not directly comparable between the different GPUs. The RTX4090 should be faster than the A6000. Our implementation is most likely slower than the official implementation at the moment.

The gradient computation kernels are currently templated to enable `float64` tensors which are required to use `torch.autograd.gradcheck`. All of the backward passes have gradcheck unit test coverage and should be computing the correct gradients for the corresponding forward pass. Additionally, the templated kernels do not allow for `float2/3/4` types which could improve performance with better memory alignment.

 The discrepancy in PSNR are most likely due to differences in the adaptive control algorithm and hyperparameter tuning. 

A comparison from one of the test images in the `garden` dataset. The official implementation image appears to be more saturated since the image is extracted from the published pdf. The branch in the exploded view and the wall is reconstructed more crisply in our implementation but the official implementation performs better on the trees and bushes.
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/2dd7f43a-ae30-46de-93f6-fc8e6e918a0d)



## Installation
This package requires CUDA which can be installed from [here](https://developer.nvidia.com/cuda-downloads). 

1. Install Python dependencies
```
pip install -r requirements.txt
```

2. Install the PyTorch CUDA extension
```
pip install -e ./
```
Note:
- Windows systems may need modify compilation flags in `setup.py`
- This step may be sensitive to the version of `pip`. This step failed after upgrading from `23.0.1` to `23.3.2`
- If `pip install` fails, this may work:
```
python setup.py build_ext && python setup.py install
```

Optional:
This project uses `clang-format` to lint the C++/CUDA files:

```
sudo apt install clang-format
```
Running `lint.sh` will run both `black` and `clang-format`.


## Verifying Install

1. Download the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset and unzip

```
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip && unzip 360_v2.zip
```

2. Update the `DATASET_PATH` in `splat_py/constants.py` to `garden` with `DOWNSAMPLE_FACTOR = 4`

3. Run `colmap_splat.py`


To run all unit tests:

```
python -m unittest discover test
```

