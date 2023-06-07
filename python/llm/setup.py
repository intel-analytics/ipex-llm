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

# >> Usage:
#
# >>>> Build for the current platform:
# python setup.py clean --all bdist_wheel
# >>>> Windows:
# python setup.py clean --all bdist_wheel --win
# >>>> Linux：
# python setup.py clean --all bdist_wheel --linux

import fnmatch
import os
import platform
import shutil
import sys
import urllib.request

from setuptools import setup

long_description = '''
    BigDL LLM
'''

exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]
BIGDL_PYTHON_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = open(os.path.join(BIGDL_PYTHON_HOME, 'version.txt'), 'r').read().strip()
llm_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
libs_dir = os.path.join(llm_home, "bigdl", "llm", "libs")


def get_llm_packages():
    llm_packages = []
    for dirpath, _, _ in os.walk(os.path.join(llm_home, "bigdl")):
        print(dirpath)
        package = dirpath.split(llm_home + os.sep)[1].replace(os.sep, '.')
        if any(fnmatch.fnmatchcase(package, pat=pattern)
               for pattern in exclude_patterns):
            print("excluding", package)
        else:
            llm_packages.append(package)
            print("including", package)
    return llm_packages


lib_urls = {}
lib_urls["Windows"] = [
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/llama.dll",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/quantize-llama.exe",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/gptneox.dll",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/quantize-gptneox.exe",
    # TODO: add bloomz
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-llama.exe",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-bloom.exe",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-gptneox.exe",
]
lib_urls["Linux"] = [
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libllama_avx2.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libllama_avx512.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/quantize-llama",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libgptneox_avx2.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libgptneox_avx512.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/quantize-gptneox",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libbloom_avx2.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/libbloom_avx512.so",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-llama",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-bloom",
    "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/main-gptneox",
]


def download_libs(url: str, change_permission=False):
    libso_file_name = url.split('/')[-1]
    libso_file = os.path.join(libs_dir, libso_file_name)
    if not os.path.exists(libso_file):
        print(">> Downloading from ", url)
        urllib.request.urlretrieve(url, libso_file)
        if change_permission:
            os.chmod(libso_file, 0o775)


def setup_package():
    package_data = {}
    package_data["Windows"] = [
        "libs/llama.dll",
        "libs/quantize-llama.exe",
        "libs/gptneox.dll",
        "libs/quantize-gptneox.exe",
        "libs/main-bloom.exe",
        "libs/main-gptneox.exe",
        "libs/main-llama.exe",
    ]
    package_data["Linux"] = [
        "libs/libllama_avx2.so",
        "libs/libllama_avx512.so",
        "libs/quantize-llama",
        "libs/libgptneox_avx2.so",
        "libs/libgptneox_avx512.so",
        "libs/quantize-gptneox",
        "libs/libbloom_avx2.so",
        "libs/libbloom_avx512.so",
        "libs/main-bloom",
        "libs/main-gptneox",
        "libs/main-llama",
    ]

    platform_name = None
    if "--win" in sys.argv:
        platform_name = "Windows"
        sys.argv.remove("--win")
    if "--linux" in sys.argv:
        platform_name = "Linux"
        sys.argv.remove("--linux")

    if platform_name is None:
        if platform.platform().startswith('Windows'):
            platform_name = "Windows"
        else:
            platform_name = "Linux"

    change_permission = True if platform_name == "Linux" else False

    # Delete legacy libs
    if os.path.exists(libs_dir):
        print(f"Deleting existing libs_dir {libs_dir} ....")
        shutil.rmtree(libs_dir)
    os.makedirs(libs_dir, exist_ok=True)

    for url in lib_urls[platform_name]:
        download_libs(url, change_permission=change_permission)

    metadata = dict(
        name='bigdl-llm',
        version=VERSION,
        description='Large Language Model Develop Toolkit',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=get_llm_packages(),
        package_dir={"": "src"},
        package_data={"bigdl.llm": package_data[platform_name]},
        include_package_data=True,
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython'],
        scripts={
            'Linux': ['src/bigdl/llm/cli/llm-cli'],
            'Windows': ['src/bigdl/llm/cli/llm-cli.ps1'],
        }[platform_name],
        platforms=['windows']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
