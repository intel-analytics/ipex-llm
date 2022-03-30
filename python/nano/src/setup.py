#!/usr/bin/env python

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import os
import fnmatch
from setuptools import setup, find_packages
import urllib.request
import os
import platform
import stat
import sys

from pathlib import Path

exclude_patterns = ["*__pycache__*", "lightning_logs", "recipe", "setup.py"]
nano_home = Path(__file__).parent.resolve()
bigdl_home = nano_home.parent.parent.parent.resolve()
VERSION = open(bigdl_home.joinpath('python', "version.txt"),
               'r').read().strip()

lib_urls = [
    "https://github.com/yangw1234/jemalloc/releases/download/v5.2.1-binary/libjemalloc.so",
    "https://github.com/leonardozcm/libjpeg-turbo/releases/download/2.1.1/libturbojpeg.so.0.2.0",
    "https://github.com/leonardozcm/tcmalloc/releases/download/v1/libtcmalloc.so"
]


def get_nano_packages():
    nano_packages = []
    for dirpath, _, _ in os.walk(nano_home.joinpath("bigdl")):
        print(dirpath)
        package = dirpath.split(str(nano_home) + os.sep)[1].replace(os.sep, '.')
        if any(fnmatch.fnmatchcase(package, pat=pattern)
                for pattern in exclude_patterns):
            print("excluding", package)
        else:
            nano_packages.append(package)
            print("including", package)
    return nano_packages


def download_libs(url: str):
    libs_dir = os.path.join(nano_home, "bigdl", "nano", "libs")
    if not os.path.exists(libs_dir):
        os.mkdir(libs_dir)
    libso_file_name = url.split('/')[-1]
    libso_file = os.path.join(libs_dir, libso_file_name)
    if not os.path.exists(libso_file):
        urllib.request.urlretrieve(url, libso_file)
    st = os.stat(libso_file)
    os.chmod(libso_file, st.st_mode | stat.S_IEXEC)


def setup_package():

    if platform.system() == "Darwin":
        plat_name = "macosx_10_11_x86_64"
    elif platform.system() == "Windows":
        plat_name = "win_amd64"
    elif platform.system() == "Linux":
        plat_name = "manylinux2010_x86_64"
    else:
        raise ValueError(f"Your platform are not supported now.")

    if plat_name != "macosx_10_11_x86_64":
        install_requires = ["intel-openmp"]
    else:
        install_requires = []

    if plat_name != "macosx_10_11_x86_64":
        tensorflow_requires = ["intel-tensorflow==2.7.0",
                               "keras==2.7.0",
                               "tensorflow-estimator==2.7.0"]
    else:
        tensorflow_requires = ["tensorflow==2.7.0",
                               "keras==2.7.0",
                               "tensorflow-estimator==2.7.0"]

    pytorch_requires = ["torch==1.9.0",
                        "torchvision==0.10.0",
                        "pytorch_lightning==1.4.2",
                        "opencv-python-headless",
                        "PyTurboJPEG",
                        "opencv-transforms",
                        "onnx",
                        "onnxruntime"]

    package_data_plat_ = {"manylinux2010_x86_64": [
        "libs/libjemalloc.so",
        "libs/libturbojpeg.so.0.2.0",
        "libs/libtcmalloc.so"
    ],
        "win_amd64": [],
        "macosx_10_11_x86_64": []}

    script_plat = {"manylinux2010_x86_64": ["../script/bigdl-nano-init"],
                   "win_amd64": ["../script/bigdl-nano-init.ps1"],
                   "macosx_10_11_x86_64": []
                   }

    if plat_name == 'manylinux2010_x86_64':
        for url in lib_urls:
            download_libs(url)

    metadata = dict(
        name='bigdl-nano',
        version=VERSION,
        description='High-performance scalable acceleration components for intel.',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        url='https://github.com/intel-analytics/BigDL',
        install_requires=install_requires,
        extras_require={"tensorflow": tensorflow_requires,
                        "pytorch": pytorch_requires},
        package_data={"bigdl.nano": package_data_plat_[plat_name]},
        scripts=script_plat[plat_name],

        packages=get_nano_packages(),
    )
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
