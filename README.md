# 3D Gaussian Splatting
A "from scratch" re-implementation of [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl and Kopanas et al.

This repository implements the forward and backwards passes using a PyTorch CUDA extension based on the algorithms descriped in the paper. Some details of the splatting and adaptive control algorithm are not explicitly described in the paper and there may be differences between this repo and the official implementation.

The forward and backward pass algorithms are detailed in MATH.md


## Performance

Evaluations done with the Mip-NeRF 360 dataset at ~1 megapixel resoloution. This corresponds to the 2x downsampled indoor scenes and 4x downsampled outdoor scenes. Every 8th image was used for the test split.


Here are some comparisons with the with the official implementation (copied from "Per-Scene Error Metrics").


| Method       | Dataset     | PSNR | SSIM | N Gaussians | Train Duration   |
|--------------|-------------|------|------|-------------|------------------|
| Official-30k | Garden 1/4x | 27.41| 0.87 |             |                  |
| Ours-30k     | Garden 1/4x | 26.80| 0.84 | 2.64M       | 22:57  (RTX4090) |
| Official-7k  | Garden 1/4x | 26.24| 0.83 |             |                  |
| Ours-7k      | Garden 1/4x | 25.77| 0.80 | 1.48M       | 3:39   (RTX4090) |
| Official-30k | Counter 1/2x| 28.70| 0.91 |             |                  |
| Ours-30k     | Counter 1/2x| 28.60| 0.90 | 2.01M       | ~26min (RTX4090) |
| Official-7k  | Counter 1/2x| 26.70| 0.87 |             |                  |
| Ours-7k      | Counter 1/2x| 27.57| 0.89 | 1.33M       | 5:03   (RTX4090) |
| Official-30k | Bonsai  1/2x| 31.98| 0.94 |             |                  |
| Ours-30k     | Bonsai  1/2x| 32.15| 0.94 | 2.60M       | 31:00  (RTX4090) |
| Official-7k  | Bonsai 1/2x | 28.85| 0.91 |             |                  |
| Ours-7k      | Bonsai 1/2x | 30.16| 0.93 | 1.87M       | 5:12   (RTX4090) |
| Official-30k | Room 1/2x   | 30.63| 0.91 |             |                  |
| Ours-30k     | Room 1/2x   | 31.68| 0.92 | 1.48M       | 23:21  (RTX4090) |
| Official-7k  | Room 1/2x   | 28.14| 0.88 |             |                  |
| Ours-7k      | Room 1/2x   | 30.28| 0.91 | 1.02M       | 3:46   (RTX4090) |


A comparison from one of the test images in the `garden` dataset. The official implementation and ground truth images appear to be more saturated since they are screen captures of the pdf.

Ours - 30k:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/075c6fd3-b92b-4733-9ac6-370a4cde8d9a)

Official Inria implementation - 30k:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/1460b7eb-a28c-43ed-b8e2-a2695f6ab805)

Ground truth:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/e3c1f0c2-3f36-41dc-8441-df856399e987)


The gradient computation kernels are currently templated to enable `float64` tensors which are required to use `torch.autograd.gradcheck`. All of the backward passes have gradcheck unit test coverage and should be computing the correct gradients for the corresponding forward pass. Additionally, the templated kernels do not allow for `float2/3/4` types which could improve performance with better memory alignment.

 The discrepancy in PSNR are most likely due to differences in the adaptive control algorithm and tuning.


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


## Training on Mip-Nerf 360 Scenes

1. Download the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset and unzip

```
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip && unzip 360_v2.zip
```


2. Run the training script:
```
python colmap_splat.py 7k --dataset_path <path to dataset> --downsample_factor 4
``` 

To run the high-quality version use `30k` instead of `7k` The `dataset_path` argument refers to the top-level folder for each dataset (`garden`, `kitchen` etc). The paper uses `--downsample_factor 4` for the outdoor scenes and `--downsample_factor 2` for the indoor scenes.


For more options:
```
python colmap_splat.py 7k --help
```

To run all unit tests:

```
python -m unittest discover test
```

