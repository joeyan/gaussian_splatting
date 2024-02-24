from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

c_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17"]
setup(
    name="splat_cuda",
    ext_modules=[
        CUDAExtension(
            name="splat_cuda",
            sources=[
                "src/bindings.cpp",
                "src/splat.cu",
                "src/projection.cu",
                "src/projection_backwards.cu",
                "src/tile_culling.cu",
            ],
        ),
    ],
    extra_compile_args={
        "cxx": c_flags,
        "nvcc": nvcc_flags,
    },
    cmdclass={"build_ext": BuildExtension},
)
