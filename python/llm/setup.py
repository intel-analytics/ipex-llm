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
from skbuild import setup

long_description = '''
    BigDL LLM
'''

exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]
BIGDL_PYTHON_HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = open(os.path.join(BIGDL_PYTHON_HOME, 'version.txt'), 'r').read().strip()
llm_home = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")


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


def setup_package():
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
        install_requires=[""],
        include_package_data=True,
        classifiers=[
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: Implementation :: CPython'],
        platforms=['linux', 'windows']
    )

    setup(**metadata)


if __name__ == '__main__':
    setup_package()
