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
from setuptools import setup
import sys

bigdl_home = os.path.abspath(__file__ + "/../../")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'),
               'r').read().strip()


def setup_package(plat_name):

    if plat_name == "win_amd64":
        platform = ['windows']
        requires = ['bigdl-nano=='+VERSION]
        dependency = []
    else:
        platform = ['mac', 'linux']
        requires = ['bigdl-orca=='+VERSION, 'bigdl-nano=='+VERSION, 'bigdl-chronos=='+VERSION,
                    'bigdl-friesian=='+VERSION, 'bigdl-serving=='+VERSION]
        dependency = [
            'https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz']

    metadata = dict(
        name='bigdl',
        version=VERSION,
        description='Building Large-Scale AI Applications for Distributed Big Data',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/analytics-zoo',
        install_requires=requires,
        dependency_links=dependency,
        include_package_data=True,
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=platform
    )

    setup(**metadata)


if __name__ == '__main__':
    idx = 0
    for arg in sys.argv:
        if arg == "--plat-name":
            break
        else:
            idx += 1
    if idx >= len(sys.argv):
        raise ValueError(
            "Cannot find --plat-name argument. bigdl-tf requires --plat-name to build.")
    verbose_plat_name = sys.argv[idx + 1]

    valid_plat_names = ("win_amd64", "manylinux2010_x86_64",
                        'macosx_10_11_x86_64')
    if verbose_plat_name not in valid_plat_names:
        raise ValueError(f"--plat-name is not valid. "
                         f"--plat-name should be one of {valid_plat_names}"
                         f" but got {verbose_plat_name}")
    plat_name = verbose_plat_name

    setup_package(plat_name)
