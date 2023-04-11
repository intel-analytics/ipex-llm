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
    "https://github.com/analytics-zoo/jemalloc/releases/download/v5.3.0/libjemalloc.dylib",
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

    # all intel-tensorflow is only avaliable for linux and windows now
    tensorflow_27_requires = ["intel-tensorflow==2.7.0; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                              platform_system!='Darwin'",
                              "tensorflow==2.7.0; platform_machine=='x86_64' and \
                              platform_system=='Darwin'"]

    tensorflow_28_requires = ["intel-tensorflow==2.8.0; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                              platform_system!='Darwin'",
                              "tensorflow==2.8.0; platform_machine=='x86_64' and \
                              platform_system=='Darwin'"]

    tensorflow_29_requires = ["intel-tensorflow==2.9.1; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                              platform_system!='Darwin'",
                              "tensorflow==2.9.0; platform_machine=='x86_64' and \
                              platform_system=='Darwin'"]

    tensorflow_210_requires = ["intel-tensorflow==2.10.0; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                               platform_system!='Darwin'",
                               "tensorflow==2.10.0; platform_machine=='x86_64' and \
                               platform_system=='Darwin'"]

    # options for stock tensorflow
    stock_tensorflow_27_requires = ["tensorflow==2.7.4; (platform_machine=='x86_64' or platform_machine == 'AMD64')"]

    stock_tensorflow_28_requires = ["tensorflow==2.8.4; (platform_machine=='x86_64' or platform_machine == 'AMD64')"]

    stock_tensorflow_29_requires = ["tensorflow==2.9.3; (platform_machine=='x86_64' or platform_machine == 'AMD64')"]

    stock_tensorflow_210_requires = ["tensorflow==2.10.1; (platform_machine=='x86_64' or platform_machine == 'AMD64')"]

    tensorflow_common_requires = ["tf2onnx==1.13.0; (platform_machine=='x86_64' or platform_machine == 'AMD64')"]

    # default tensorflow_dep
    tensorflow_requires = tensorflow_29_requires + tensorflow_common_requires
    tensorflow_210_requires += tensorflow_common_requires
    tensorflow_29_requires += tensorflow_common_requires
    tensorflow_28_requires += tensorflow_common_requires
    tensorflow_27_requires += tensorflow_common_requires
    stock_tensorflow_27_requires += tensorflow_common_requires
    stock_tensorflow_28_requires += tensorflow_common_requires
    stock_tensorflow_29_requires += tensorflow_common_requires
    stock_tensorflow_210_requires += tensorflow_common_requires

    # ipex is only avaliable for linux now
    pytorch_113_requires = ["torch==1.13.1",
                            "torchvision==0.14.1",
                            "intel_extension_for_pytorch==1.13.100;platform_system=='Linux'"]

    # This is for xpu support (currently we only support 1.13)
    # should be installed with -f https://developer.intel.com/ipex-whl-stable-xpu
    pytorch_113_xpu_requires = ["torch==1.13.0a0",
                                "torchvision==0.14.1a0",
                                "intel_extension_for_pytorch==1.13.10+xpu;platform_system=='Linux'"]

    pytorch_112_requires = ["torch==1.12.1",
                            "torchvision==0.13.1",
                            "intel_extension_for_pytorch==1.12.300;platform_system=='Linux'"]

    pytorch_111_requires = ["torch==1.11.0",
                            "torchvision==0.12.0",
                            "intel_extension_for_pytorch==1.11.0;platform_system=='Linux'"]

    pytorch_110_requires = ["torch==1.10.1",
                            "torchvision==0.11.2",
                            "intel_extension_for_pytorch==1.10.100;platform_system=='Linux'"]

    # this require install option --extra-index-url https://download.pytorch.org/whl/nightly/
    pytorch_nightly_requires = ["torch~=1.14.0.dev",
                                "torchvision~=0.15.0.dev"]

    pytorch_common_requires = ["pytorch_lightning==1.6.4",
                               "torchmetrics==0.11.0",
                               "opencv-python-headless",
                               "PyTurboJPEG",
                               "opencv-transforms",
                               "cryptography==39.0.1"]

    # default pytorch_dep
    pytorch_requires = pytorch_113_requires + pytorch_common_requires
    pytorch_113_requires += pytorch_common_requires
    pytorch_113_xpu_requires += pytorch_common_requires
    pytorch_112_requires += pytorch_common_requires
    pytorch_111_requires += pytorch_common_requires
    pytorch_110_requires += pytorch_common_requires
    pytorch_nightly_requires += pytorch_common_requires

    inference_requires = ["onnx==1.12.0",
                          "onnxruntime==1.12.1",
                          "onnxruntime-extensions==0.4.2; platform_system!='Darwin'",
                          "onnxruntime-extensions==0.3.1; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                          platform_system=='Darwin'",
                          "openvino-dev==2022.3.0",
                          "neural-compressor==2.0; platform_system!='Windows'",
                          "onnxsim==0.4.8; platform_system!='Darwin'",
                          "onnxsim==0.4.1; (platform_machine=='x86_64' or platform_machine == 'AMD64') and \
                          platform_system=='Darwin'"]

    install_requires = ["intel-openmp; (platform_machine=='x86_64' or platform_machine == 'AMD64')",
                        "cloudpickle",
                        "protobuf==3.19.5",
                        "py-cpuinfo",
                        "pyyaml",
                        "packaging",
                        "sigfig",
                        "setuptools<66"]

    package_data = [
        "libs/libjemalloc.so",
        "libs/libturbojpeg.so.0.2.0",
        "libs/libtcmalloc.so"
    ]

    for url in lib_urls:
        download_libs(url)

    scripts = ["scripts/bigdl-nano-init",
               "scripts/bigdl-nano-init.ps1",
               "scripts/bigdl-nano-unset-env",
               "scripts/bigdl-nano-unset-env.ps1"]

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
                        "tensorflow_27": tensorflow_27_requires,
                        "tensorflow_28": tensorflow_28_requires,
                        "tensorflow_29": tensorflow_29_requires,
                        "tensorflow_210": tensorflow_210_requires,
                        "stock_tensorflow_27": stock_tensorflow_27_requires,
                        "stock_tensorflow_28": stock_tensorflow_28_requires,
                        "stock_tensorflow_29": stock_tensorflow_29_requires,
                        "stock_tensorflow_210": stock_tensorflow_210_requires,
                        "pytorch": pytorch_requires,
                        "pytorch_113": pytorch_113_requires,
                        "pytorch_112": pytorch_112_requires,
                        "pytorch_111": pytorch_111_requires,
                        "pytorch_110": pytorch_110_requires,
                        "pytorch_113_xpu": pytorch_113_xpu_requires,
                        "pytorch_nightly": pytorch_nightly_requires,
                        "inference": inference_requires},
        package_data={"bigdl.nano": package_data},
        scripts=scripts,
        package_dir={"": "src"},
        entry_points = {
            'console_scripts': ['bigdl-submit=bigdl.nano.k8s:main'],
            },
        packages=get_nano_packages(),
    )
    setup(**metadata)


if __name__ == '__main__':
    setup_package()
