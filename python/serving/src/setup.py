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
from shutil import copytree, rmtree
from setuptools import setup
import fnmatch

bigdl_home = os.path.abspath(__file__ + "/../../../..")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()

SCRIPTS_TARGET = os.path.join(bigdl_home, "scala/serving/scripts/")
TMP_PATH = "bigdl/share/serving"
if os.path.exists(TMP_PATH):
    rmtree(TMP_PATH)
copytree(SCRIPTS_TARGET, TMP_PATH)


def get_bigdl_packages():
    bigdl_python_home = os.path.abspath(__file__ + "/..")
    bigdl_packages = ['bigdl.share.serving']
    source_dir = os.path.join(bigdl_python_home, "bigdl")
    for dirpath, dirs, files in os.walk(source_dir):
        package = dirpath.split(bigdl_python_home)[1].replace('/', '.')
        if any(fnmatch.fnmatchcase(package, pat=pattern)
                for pattern in exclude_patterns):
            print("excluding", package)
        else:
            bigdl_packages.append(package)
            print("including", package)
    return bigdl_packages


def setup_package():
    script_names = [f for f in os.listdir(SCRIPTS_TARGET) if
                    os.path.isfile(os.path.join(SCRIPTS_TARGET, f))]
    scripts = list(map(lambda script: os.path.join(
        SCRIPTS_TARGET, script), script_names))
    print(os.listdir("../.."))
    metadata = dict(
        name='bigdl-serving',
        version=VERSION,
        description='Distributed and Automated Model Inference on Big Data Streaming Frameworks',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=get_bigdl_packages(),
        # package_dir={'bigdl': '../../serving/src/bigdl'},
        # package_data={"bigdl.conf": ['config.yaml']},
        include_package_data=False,
        scripts=scripts,
        install_requires=['redis', 'pyyaml', 'httpx', 'pyarrow', 'opencv-python', 'kafka-python'],
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['mac', 'linux']
    )

    setup(**metadata)


if __name__ == '__main__':
    try:
        setup_package()
    except Exception as e:
        raise e
