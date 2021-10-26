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
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

bigdl_home = os.path.abspath(__file__ + "/../../../..")
VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()

plat_name = "linux-x86_64"

class bdist_wheel(_bdist_wheel):
    def run(self):
        plat_name = self.plat_name
        _bdist_wheel.run(self)


def setup_package():
    package_data_plat_ = {"linux-x86_64":["libiomp5.so", "libmklml_intel.so"],
                          "darwin-x86_64":["libiomp5.dylib", "libmklml.dylib"]}

    packages_name = "bigdl.share.tflibs." + plat_name

    metadata = dict(
        name='bigdl-math',
        version=VERSION,
        cmdclass={
          'bdist_wheel': bdist_wheel
        },
        description='Math Dependency Library for bigdl-orca',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/BigDL',
        packages=[packages_name],
        package_data={packages_name: package_data_plat_[plat_name]}
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
