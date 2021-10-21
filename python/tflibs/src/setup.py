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

bigdl_home = os.path.abspath(__file__ + "/../../../..")
VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()

def setup_package():
    metadata = dict(
        name='bigdl-tf',
        version=VERSION,
        description='TensorFlow Dependency Library for bigdl-orca',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        packages=["bigdl.share.tflibs.linux-x86_64"],
        url='https://github.com/intel-analytics/BigDL',
        package_data={"bigdl.share.tflibs.linux-x86_64": ["libtensorflow_framework-zoo.so",
                                                          "libtensorflow_jni.so"]}
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
