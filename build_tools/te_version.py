# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from pathlib import Path
import subprocess

DAS_VERSION="1.6"

def abi_value():
    try:
        return (
            subprocess.check_output("echo '#include <string>' | gcc -x c++ -E -dM - | fgrep _GLIBCXX_USE_CXX11_ABI", shell=True)
            .decode('ascii')
            .strip()[-1]
        )
    except Exception:
        return abiUNKNOWN

def dtk_version_value():
    try:
       dtk_path=os.getenv('ROCM_PATH')
       dtk_version_path = os.path.join(dtk_path, '.info', "version-dev")
       with open(dtk_version_path, 'r',encoding='utf-8') as file:
           lines = file.readlines()
       dtk_version="dtk"+lines[0][:].replace(".", "")
       return dtk_version
    except Exception:
       return UNKNOWN

def te_version() -> str:
    """Transformer Engine version string

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    root_path = Path(__file__).resolve().parent
    with open(root_path / "VERSION.txt", "r") as f:
        version = f.readline().strip()
    if not int(os.getenv("NVTE_NO_LOCAL_VERSION", "0")) and not bool(
        int(os.getenv("NVTE_RELEASE_BUILD", "0"))
    ):
        try:
            output = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                cwd=root_path,
                check=True,
                universal_newlines=True,
            )
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            commit = output.stdout.strip()
            version += "+das"+ DAS_VERSION + f".git{commit}"+ ".abi"+str(abi_value()) + "." +str(dtk_version_value())
    else:
        version += "+das"+ DAS_VERSION + f".opt1"+ "." +str(dtk_version_value())
    return version
