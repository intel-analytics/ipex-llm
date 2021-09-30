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
from setuptools import setup
import urllib.request
import os
import stat
import sys
import re
from html.parser import HTMLParser


exclude_patterns = ["*__pycache__*", "lightning_logs", "recipe", "setup.py"]
nano_home = os.path.abspath(__file__ + "/../src/")


def get_nano_packages():
    nano_packages = []
    for dirpath, _, _ in os.walk(nano_home + "/bigdl"):
        print(dirpath)
        package = dirpath.split(nano_home + "/")[1].replace('/', '.')
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


class URLHtmlParser(HTMLParser):

    def __init__(self):
        super().__init__()
        self.links = {}
        self.unmatched_link = None

    def handle_starttag(self, tag, attrs):
        if tag != 'a':
            return

        for attr in attrs:
            if 'href' in attr[0]:
                self.unmatched_link = attr[1]
                break

    def handle_data(self, data):
        if self.unmatched_link is not None:
            self.links[data] = self.unmatched_link
            self.unmatched_link = None


def parse_find_index_page(url):
    with urllib.request.urlopen(url, timeout=30) as f:
        content = f.read()
    content = content.decode('utf8')
    parser = URLHtmlParser()
    parser.feed(content)
    return parser.links


def setup_package():

    py_version = sys.version_info

    assert py_version.major == 3, "Nano only supports python3.x"

    ipex_version = "1.8.0"
    ipex_version_major = "1.8"
    ipex_whl_name = f"torch_ipex-{ipex_version}-cp{py_version.major}{py_version.minor}" \
                    f"-cp{py_version.major}{py_version.minor}m-linux_x86_64.whl"
    ipex_link = f"https://intel-optimized-pytorch.s3.cn-north-1.amazonaws.com.cn/wheels/" \
                f"v{ipex_version_major}/{ipex_whl_name}"

    torch_links = parse_find_index_page(
        "https://download.pytorch.org/whl/torch_stable.html")
    torchvision_version = "0.9.0"
    pytorch_version = "1.8.0"
    pytorch_url_suffix = "m-linux_x86_64.whl" if py_version.minor < 8 else "-linux_x86_64.whl"

    torchvision_whl_name = f"cpu/torchvision-{torchvision_version}%2Bcpu-cp{py_version.major}{py_version.minor}" \
                           f"-cp{py_version.major}{py_version.minor}" + pytorch_url_suffix
    pytorch_whl_name = f"cpu/torch-{pytorch_version}%2Bcpu-cp{py_version.major}{py_version.minor}" \
                       f"-cp{py_version.major}{py_version.minor}" + pytorch_url_suffix

    # both pytorch_lightning and ipex depends on pytorch
    # listing it here to make sure installing the correct version
    pytorch_link = "https://download.pytorch.org/whl/" + \
        torch_links[pytorch_whl_name]
    torchvision_link = "https://download.pytorch.org/whl/" + \
        torch_links[torchvision_whl_name]

    install_requires = ["intel-openmp"]

    tensorflow_requires = ["intel-tensorflow"]

    pytorch_requires = ["pytorch_lightning",
                        "opencv-python-headless",
                        "PyTurboJPEG",
                        "opencv-transforms",
                        f"torch_ipex @ {ipex_link}",
                        f"torchvision @ {torchvision_link}"]

    lib_urls = [
        "https://github.com/yangw1234/jemalloc/releases/download/v5.2.1-binary/libjemalloc.so",
        "https://github.com/leonardozcm/libjpeg-turbo/releases/download/2.1.1/libturbojpeg.so.0.2.0",
        "https://github.com/leonardozcm/tcmalloc/releases/download/v1/libtcmalloc.so"
    ]
    for url in lib_urls:
        download_libs(url)

    metadata = dict(
        name='bigdl-nano',
        version='0.14.0.dev0',
        description='',
        author='',
        author_email='',
        url='https://github.com/intel-analytics/analytics-zoo/tree/bigdl-2.0',
        install_requires=install_requires,
        extras_require={"tensorflow": tensorflow_requires,
                        "pytorch": pytorch_requires},
        packages=get_nano_packages(),
        package_data={"bigdl.nano": [
            "libs/libjemalloc.so", "libs/libturbojpeg.so.0.2.0", "libs/libtcmalloc.so"]},
        package_dir={'': 'src'},
        scripts=['script/bigdl-nano-init']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
