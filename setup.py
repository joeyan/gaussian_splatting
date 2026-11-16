import os
import shutil
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

c_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17"]

# On Windows with HIP, MSVC-compiled bindings.cpp cannot link c10::ValueError
# from the clang-built c10.dll (inherited ctor absent from the import lib).
# Copy bindings.cpp to bindings_winhip.cu so BuildExtension routes it through
# hipcc/amdclang++, which uses the same ABI as c10.dll.
import torch
if os.name == 'nt' and torch.version.hip is not None:
    _here = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(os.path.join(_here, "src", "bindings.cpp"),
                 os.path.join(_here, "src", "bindings_winhip.cu"))
    _bindings = "src/bindings_winhip.cu"
else:
    _bindings = "src/bindings.cpp"

setup(
    name="splat_cuda",
    ext_modules=[
        CUDAExtension(
            name="splat_cuda",
            sources=[
                _bindings,
                "src/depth.cu",
                "src/precompute_sh.cu",
                "src/projection.cu",
                "src/projection_backward.cu",
                "src/render.cu",
                "src/render_backward.cu",
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
