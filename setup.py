import os
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def main():
    curr_dir = Path(__file__).absolute().parent
    # Try to locate torch shared libraries directory
    try:
        import torch  # noqa: F401
        torch_dir = Path(torch.__file__).parent
        torch_lib_dir = torch_dir / "lib"
    except Exception:
        torch_lib_dir = None

    setup(
        ext_modules=[
            CUDAExtension(
                name="atlas._c",
                sources=[
                    "lib/bind.cpp",
                    # "lib/cache.cpp",
                    # "lib/dedup.cpp",
                    "lib/sampler.cpp",
                    "lib/fixed_sampler.cpp",
                    "lib/tcsr.cpp",
                    "lib/utils.cpp",
                    "lib/recover_kernel.cu",
                ],
                include_dirs=[curr_dir/"include/"],
                extra_compile_args={
                    "cxx": ["-std=c++17", "-fopenmp", "-O3", "-mavx512f"],
                    "nvcc": ["-O3"],
                },
                library_dirs=[str(torch_lib_dir)] if torch_lib_dir else [],
                runtime_library_dirs=["$ORIGIN/../torch/lib"] if os.name == "posix" else [],
                extra_link_args=(
                    [
                        "-fopenmp",
                        "-Wl,-rpath,$ORIGIN/../torch/lib",
                    ]
                    + ([f"-Wl,-rpath,{torch_lib_dir}"] if torch_lib_dir else [])
                )
            )],
        cmdclass={
            "build_ext": BuildExtension
        })


if __name__ == "__main__":
    main()
