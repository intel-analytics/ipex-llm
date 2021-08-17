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

exclude_patterns = ["*__pycache__*", "lightning_logs", "recipe", "setup.py"]
nano_home = os.path.abspath(__file__ + "/../src/")
print(nano_home)


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
    

def load_requirements(file_name="requirements.txt", comment_char='#'):
    """
    Load requirements from a file,
    return install_requires_list
    """
    with open(file_name, 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    _install_requires_list = []
    _dependency_links_tmp = ""
    for ln in lines:
        # Filter out urls
        if comment_char in ln:
            text_start = 0
            for s_char in ln:
                if s_char is comment_char:
                    text_start += 1
            ln = ln[text_start:].strip()
            if ln.startswith('http'):
                _dependency_links_tmp = ln
            else:
                _dependency_links_tmp = ""
        elif ln:
            ln = ln + " @ " + _dependency_links_tmp if _dependency_links_tmp else ln
            _dependency_links_tmp = ""
            _install_requires_list.append(ln)
        else:
            _dependency_links_tmp = ""
    return _install_requires_list


def setup_package():
    install_requires_list = load_requirements()

    metadata = dict(
        name='bigdl-nano',
        version='0.0.1.dev0',
        description='',
        author='',
        author_email='',
        url='https://github.com/intel-analytics/analytics-zoo/tree/bigdl-2.0',
        install_requires=install_requires_list,
        packages=get_nano_packages(),
        package_dir = {'': 'src'},
        scripts=['script/bigdl-nano-run']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
