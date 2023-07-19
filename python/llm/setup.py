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
import requests
import re
import glob

from setuptools import setup

long_description = '''
    BigDL LLM
'''

exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]
BIGDL_PYTHON_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = open(os.path.join(BIGDL_PYTHON_HOME, 'version.txt'), 'r').read().strip()
llm_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
github_artifact_dir = os.path.join(llm_home, '../llm-binary')
libs_dir = os.path.join(llm_home, "bigdl", "llm", "libs")
CONVERT_DEP = ['numpy >= 1.22', 'torch', 'transformers', 'sentencepiece', 'accelerate']


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


def obtain_lib_urls():
    base_url = "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/"
    windows_binarys = ["llama.dll", "gptneox.dll", "bloom.dll",
                       "quantize-llama.exe", "quantize-gptneox.exe", "quantize-bloom.exe",
                       "main-llama.exe", "main-gptneox.exe", "main-bloom.exe",
                       "starcoder.dll", "quantize-starcoder.exe", "main-starcoder.exe",
                       "libllama_vnni.dll", "libgptneox_vnni.dll", "libbloom_vnni.dll",
                       "quantize-llama_vnni.exe", "quantize-gptneox_vnni.exe", "quantize-bloom_vnni.exe",
                       "main-llama_vnni.exe", "main-gptneox_vnni.exe", "main-bloom_vnni.exe",
                       "libstarcoder_vnni.dll", "quantize-starcoder_vnni.exe", "main-starcoder_vnni.exe"]
    linux_binarys = ["libllama_avx2.so", "libgptneox_avx2.so", "libbloom_avx2.so",
                     "libllama_avx512.so", "libgptneox_avx512.so", "libbloom_avx512.so",
                     "quantize-llama", "quantize-gptneox", "quantize-bloom",
                     "main-llama_avx2", "main-gptneox_avx2", "main-bloom_avx2",
                     "main-llama_avx512", "main-gptneox_avx512", "main-bloom_avx512",
                     "libstarcoder_avx512.so", "main-starcoder_avx512", "quantize-starcoder",
                     "libstarcoder_avx2.so", "main-starcoder_avx2"]

    def get_date_urls(base_url):
        # obtain all urls based on date(format: xxxx-xx-xx)
        text = ''
        try:
            text = requests.get(base_url).text
        except Exception as e:
            print("error - > ",base_url,e)
            pass
        reg = "https://sourceforge.net/projects/analytics-zoo/files/bigdl-llm/[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}/"
        urls =  re.findall(reg, text)
        return urls

    def get_urls_for_binary(date_urls, binarys):
        # Sort by time from near to far
        date_urls = sorted(date_urls, reverse=True)
        binary_url = {}
        download_num = len(binarys)
        for url in date_urls:
            try:
                text = requests.get(url).text
            except Exception as e:
                print("error - > ", url, e)
                continue
            for binary in binarys:
                if binary in binary_url:
                    continue
                # Filename hard matching
                match_pattern = "\"name\":\"{}\"".format(binary)
                if match_pattern in text:
                    lib_url = url + binary
                    binary_url[binary] = lib_url
                    download_num -= 1
                    if download_num == 0:
                        break
            if download_num == 0:
                break
        return binary_url

    lib_urls = {}
    date_urls = get_date_urls(base_url)
    windows_binary_urls = get_urls_for_binary(date_urls, windows_binarys)
    lib_urls["Windows"] = list(windows_binary_urls.values())
    linux_binary_urls = get_urls_for_binary(date_urls, linux_binarys)
    lib_urls["Linux"] = list(linux_binary_urls.values())
    return lib_urls


def download_libs(url: str, change_permission=False):
    libso_file_name = url.split('/')[-1]
    libso_file = os.path.join(libs_dir, libso_file_name)
    if not os.path.exists(libso_file):
        print(">> Downloading from ", url)
        urllib.request.urlretrieve(url, libso_file)
    else:
        print('>> Skip downloading ', libso_file)
    if change_permission:
        os.chmod(libso_file, 0o775)


def setup_package():
    package_data = {}
    package_data["Windows"] = [
        "libs/llama.dll",
        "libs/gptneox.dll",
        "libs/bloom.dll",
        "libs/starcoder.dll",
        "libs/quantize-llama.exe",
        "libs/quantize-gptneox.exe",
        "libs/quantize-bloom.exe",
        "libs/quantize-starcoder.exe",
        "libs/main-bloom.exe",
        "libs/main-gptneox.exe",
        "libs/main-llama.exe",
        "libs/main-starcoder.exe",
        "libs/libllama_vnni.dll", 
        "libs/libgptneox_vnni.dll", 
        "libs/libbloom_vnni.dll",
        "libs/libstarcoder_vnni.dll", 
        "libs/quantize-llama_vnni.exe",
        "libs/quantize-gptneox_vnni.exe", 
        "libs/quantize-bloom_vnni.exe",
        "libs/quantize-starcoder_vnni.exe", 
        "libs/main-llama_vnni.exe", 
        "libs/main-gptneox_vnni.exe", 
        "libs/main-bloom_vnni.exe",
        "libs/main-starcoder_vnni.exe"
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
        "libs/quantize-bloom",
        "libs/libstarcoder_avx512.so",
        "libs/libstarcoder_avx2.so",
        "libs/quantize-starcoder",
        "libs/main-bloom_avx2",
        "libs/main-bloom_avx512",
        "libs/main-gptneox_avx2",
        "libs/main-gptneox_avx512",
        "libs/main-llama_avx2",
        "libs/main-llama_avx512",
        "libs/main-starcoder_avx512",
        "libs/main-starcoder_avx2",
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
    
    # copy built files for github workflow
    for built_file in glob.glob(os.path.join(github_artifact_dir,'*')):
        print(f'Copy workflow built file: {built_file}')
        shutil.copy(built_file, libs_dir)

    lib_urls = obtain_lib_urls()

    for url in lib_urls[platform_name]:
        download_libs(url, change_permission=change_permission)

    all_requires = ['py-cpuinfo']
    all_requires += CONVERT_DEP

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
        package_data={"bigdl.llm": package_data[platform_name] + ["cli/prompts/*.txt"]},
        include_package_data=True,
        entry_points={
            "console_scripts": [
                'llm-convert=bigdl.llm.convert_model:main'
            ]
        },
        extras_require={"all": all_requires},
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython'],
        scripts={
            'Linux': ['src/bigdl/llm/cli/llm-cli', 'src/bigdl/llm/cli/llm-chat'],
            'Windows': ['src/bigdl/llm/cli/llm-cli.ps1', 'src/bigdl/llm/cli/llm-chat.ps1'],
        }[platform_name],
        platforms=['windows']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
