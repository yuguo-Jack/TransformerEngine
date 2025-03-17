# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""PyTorch related extensions."""
import os
from pathlib import Path

import setuptools

from .utils import (
    rocm_build,
    hipify,
    all_files_in_dir,
    cuda_archs,
    cuda_version,
)


def setup_pytorch_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for PyTorch support"""

    # Source files
    csrc_source_files = Path(csrc_source_files)
    extensions_dir = csrc_source_files / "extensions"
    sources = [
        csrc_source_files / "common.cpp",
    ] + all_files_in_dir(extensions_dir)

    # Header files
    include_dirs = [
        common_header_files,
        common_header_files / "common",
        common_header_files / "common" / "include",
        csrc_header_files,
    ]

    if rocm_build():
        current_file_path = Path(__file__).parent.resolve()
        base_dir = current_file_path.parent
        sources = hipify(base_dir, csrc_source_files, sources, include_dirs)

    # Compiler flags
    cxx_flags = [
        "-O3",
        "-fvisibility=hidden",
    ]
    if rocm_build():
        nvcc_flags = [
            "-O3",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT16_OPERATORS__",
            "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
            "-U__HIP_NO_BFLOAT162_OPERATORS__",
            "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
        ]
    else:
        nvcc_flags = [
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ]
    # Version-dependent CUDA options
    if rocm_build():
        ##TODO: Figure out which hipcc version starts to support this parallel compilation
        nvcc_flags.extend(["-parallel-jobs=4"])
    else:
        cuda_architectures = cuda_archs()
        if "70" in cuda_architectures:
            nvcc_flags.extend(["-gencode", "arch=compute_70,code=sm_70"])
        try:
            version = cuda_version()
        except FileNotFoundError:
            print("Could not determine CUDA Toolkit version")
        else:
            if version < (12, 0):
                raise RuntimeError("Transformer Engine requires CUDA 12.0 or newer")
            nvcc_flags.extend(
                (
                    "--threads",
                    os.getenv("NVTE_BUILD_THREADS_PER_JOB", "1"),
                )
            )
    
            for arch in cuda_architectures.split(";"):
                if arch == "70":
                    continue  # Already handled
                nvcc_flags.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])

    # Libraries
    library_dirs = []
    libraries = []
    if bool(int(os.getenv("NVTE_UB_WITH_MPI", "0"))):
        assert (
            os.getenv("MPI_HOME") is not None
        ), "MPI_HOME=/path/to/mpi must be set when compiling with NVTE_UB_WITH_MPI=1!"
        mpi_path = Path(os.getenv("MPI_HOME"))
        include_dirs.append(mpi_path / "include")
        cxx_flags.append("-DNVTE_UB_WITH_MPI")
        nvcc_flags.append("-DNVTE_UB_WITH_MPI")
        library_dirs.append(mpi_path / "lib")
        libraries.append("mpi")

    # Construct PyTorch CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from torch.utils.cpp_extension import CUDAExtension

    return CUDAExtension(
        name="transformer_engine_torch",
        sources=[str(src) for src in sources],
        include_dirs=[str(inc) for inc in include_dirs],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
        libraries=[str(lib) for lib in libraries],
        library_dirs=[str(lib_dir) for lib_dir in library_dirs],
    )
