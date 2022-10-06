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


import fnmatch
from setuptools import setup
import urllib.request
import os
import stat

long_description = '''
BigDL Nano automatically accelerates TensorFlow and PyTorch pipelines 
by applying modern CPU optimizations.

See [here](https://bigdl.readthedocs.io/en/latest/doc/Nano/Overview/nano.html) 
for more information.
'''

exclude_patterns = ["*__pycache__*", "lightning_logs", "recipe", "setup.py"]
nano_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")

BIGDL_PYTHON_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = open(os.path.join(BIGDL_PYTHON_HOME, 'version.txt'), 'r').read().strip()


lib_urls = [
    "https://github.com/analytics-zoo/jemalloc/releases/download/v5.3.0/libjemalloc.so",
    "https://github.com/analytics-zoo/libjpeg-turbo/releases/download/v2.1.4/libturbojpeg.so.0.2.0",
    "https://github.com/analytics-zoo/tcmalloc/releases/download/v1/libtcmalloc.so"
]


def get_nano_packages():
    nano_packages = []
    for dirpath, _, _ in os.walk(os.path.join(nano_home, "bigdl")):
        print(dirpath)
        package = dirpath.split(nano_home + os.sep)[1].replace(os.sep, '.')
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
        os.makedirs(libs_dir, exist_ok=True)
    libso_file_name = url.split('/')[-1]
    libso_file = os.path.join(libs_dir, libso_file_name)
    if not os.path.exists(libso_file):
        urllib.request.urlretrieve(url, libso_file)
    st = os.stat(libso_file)
    os.chmod(libso_file, st.st_mode | stat.S_IEXEC)


def setup_package():

    tensorflow_requires = ["intel-tensorflow==2.7.0",
                           "keras==2.7.0",
                           "tensorflow-estimator==2.7.0"]

    pytorch_requires = ["torch==1.11.0",
                        "torchvision==0.12.0",
                        "pytorch_lightning==1.6.4",
                        "torchmetrics==0.7.2",
                        "opencv-python-headless",
                        "PyTurboJPEG",
                        "opencv-transforms",
                        "intel_extension_for_pytorch==1.11.0"]

    install_requires = ["intel-openmp", "cloudpickle", "protobuf==3.19.4"]

    package_data = [
        "libs/libjemalloc.so",
        "libs/libturbojpeg.so.0.2.0",
        "libs/libtcmalloc.so"
    ]

    for url in lib_urls:
        download_libs(url)

    scripts = ["scripts/bigdl-nano-init",
               "scripts/bigdl-nano-init.ps1",
               "scripts/bigdl-nano-unset-env"]

    metadata = dict(
        name='bigdl-nano',
        version=VERSION,
        description='High-performance scalable acceleration components for intel.',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        url='https://github.com/intel-analytics/BigDL',
        install_requires=install_requires,
        extras_require={"tensorflow": tensorflow_requires,
                        "pytorch": pytorch_requires},
        package_data={"bigdl.nano": package_data},
        scripts=scripts,
        package_dir={"": "src"},
        packages=get_nano_packages(),
    )
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
