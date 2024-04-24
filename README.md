# 3D Gaussian Splatting
A "from scratch" re-implementation of [3D Gaussian Splatting
for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) by Kerbl and Kopanas et al.

This repository implements the forward and backwards passes using a PyTorch CUDA extension based on the algorithms descriped in the paper. Some details of the splatting and adaptive control algorithm are not explicitly described in the paper and there may be differences between this repo and the official implementation.

## Motivation

1. Provide a detailed explanation of the differential rasterization algorithm. The forward and backward pass are detailed in [MATH.md](/MATH.md)
2. Permissive license. The original implementation does not allow commercial use and was never referenced during the development of this repository.
3. Modular projection functions and gradient checks allow for easier experimentation with camera/pose gradients, new camera models etc. 
4. Minimal dependencies.

If there are any issues/errors please open an Issue or Pull Request!

## Performance

Evaluations done with the Mip-NeRF 360 dataset at ~1 megapixel resoloution. This corresponds to the 2x downsampled indoor scenes and 4x downsampled outdoor scenes. Every 8th image was used for the test split. Here are some comparisons with the with the official Inria implementation (copied from "Per-Scene Error Metrics").


| Method    | Dataset     | PSNR | SSIM | N Gaussians | Train Duration   |
|-----------|-------------|------|------|-------------|------------------|
| Inria-30k | Garden 1/4x | 27.41| 0.87 |             |                  |
| Ours-30k  | Garden 1/4x | 27.05| 0.85 | 2.86M       | 20:18  (RTX4090) |
| Inria-7k  | Garden 1/4x | 26.24| 0.83 |             |                  |
| Ours-7k   | Garden 1/4x | 25.83| 0.80 | 1.52M       | 3:05   (RTX4090) |
| Inria-30k | Counter 1/2x| 28.70| 0.91 |             |                  |
| Ours-30k  | Counter 1/2x| 28.75| 0.90 | 1.84M       | 23:37  (RTX4090) |
| Inria-7k  | Counter 1/2x| 26.70| 0.87 |             |                  |
| Ours-7k   | Counter 1/2x| 27.59| 0.89 | 1.37M       | 4:10   (RTX4090) |
| Inria-30k | Bonsai  1/2x| 31.98| 0.94 |             |                  |
| Ours-30k  | Bonsai  1/2x| 32.21| 0.95 | 2.85M       | 27:22  (RTX4090) |
| Inria-7k  | Bonsai 1/2x | 28.85| 0.91 |             |                  |
| Ours-7k   | Bonsai 1/2x | 30.42| 0.93 | 1.86M       | 4:19   (RTX4090) |
| Inria-30k | Room 1/2x   | 30.63| 0.91 |             |                  |
| Ours-30k  | Room 1/2x   | 31.73| 0.93 | 1.53M       | 20:13  (RTX4090) |
| Inria-7k  | Room 1/2x   | 28.14| 0.88 |             |                  |
| Ours-7k   | Room 1/2x   | 30.30| 0.91 | 1.01M       | 3:17   (RTX4090) |


A comparison from one of the test images in the `garden` dataset. The official implementation and ground truth images appear to be more saturated since they are screen captures of the pdf.

Ours - 30k:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/519a5f04-82f3-4291-b063-c122efd22c19)

Official Inria implementation - 30k:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/1460b7eb-a28c-43ed-b8e2-a2695f6ab805)

Ground truth:
![image](https://github.com/joeyan/gaussian_splatting/assets/17635504/e3c1f0c2-3f36-41dc-8441-df856399e987)


## Installation
This package requires CUDA which can be installed from [here](https://developer.nvidia.com/cuda-downloads). 

1. Install Python dependencies
```
pip install -r requirements.txt
```

2. Install the PyTorch CUDA extension
```
python setup.py build_ext && python setup.py install
```
Note:
- Windows systems may need modify compilation flags in `setup.py`

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

## References

The original paper:
```
@Article{kerbl3Dgaussians,
      author = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal = {ACM Transactions on Graphics},
      number = {4},
      volume = {42},
      month = {July},
      year = {2023},
      url= {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```

The EWA Splatting approach that is the basis for 3D Gaussian Splatting:
```
@Article{zwicker2002ewa,
    author={M. Zwicker and H. Pfister and J. van Baar and M. Gross},
    title={EWA Splatting},
    journal={IEEE Transactions on Visualization and Computer Graphics},
    number={3},
    volume={8},
    month={July},
    year={2002},
    publisher={IEEE},
    url={https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf}
}
```

`gsplat` [Mathematical Supplement](https://arxiv.org/abs/2312.02121)
```
@misc{ye2023mathematical,
    title={Mathematical Supplement for the $\texttt{gsplat}$ Library}, 
    author={Vickie Ye and Angjoo Kanazawa},
    year={2023},
    eprint={2312.02121},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

A great reference for matrix derivatives:
```
@misc{giles2008extended,
    title={An extended collection of matrix derivative results for forward and reverse mode algorithmic differentiation}, 
    author={Mike Giles},
    month={January}
    year={2008},
    url={https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf}
}
```
