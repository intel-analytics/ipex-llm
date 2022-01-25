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

bigdl_home = os.path.abspath(__file__ + "/../../")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()


def setup_package():
    metadata = dict(
        name='bigdl',
        version=VERSION,
        description='Building Large-Scale AI Applications for Distributed Big Data',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        install_requires=['bigdl-orca=='+VERSION, 'bigdl-nano=='+VERSION, 'bigdl-chronos=='+VERSION,
                          'bigdl-friesian=='+VERSION, 'bigdl-serving=='+VERSION],
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
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
    setup_package()
