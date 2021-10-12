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
import sys
from shutil import copytree, rmtree
import fnmatch
from setuptools import setup

TEMP_PATH = "bigdl/share/friesian"
bigdl_home = os.path.abspath(__file__ + "/../../../..")
exclude_patterns = ["*__pycache__*", "*ipynb_checkpoints*"]

VERSION = open(os.path.join(bigdl_home, 'python/version.txt'), 'r').read().strip()

building_error_msg = """
If you are packing python API from BigDL source, you must build BigDL first
and run sdist.
    To build BigDL with maven you can run:
      cd $BigDL_HOME
      ./make-dist.sh
    Building the source dist is done in the Python directory:
      cd python
      python setup.py sdist
      pip install dist/*.tar.gz"""


def build_from_source():
    code_path = bigdl_home + "/python/friesian/src/bigdl/friesian/__init__.py"
    print("Checking: %s to see if build from source" % code_path)
    if os.path.exists(code_path):
        return True
    return False


def init_env():
    if build_from_source():
        print("Start to build distributed package")
        print("HOME OF BIGDL: " + bigdl_home)
        dist_source = bigdl_home + "/dist"
        if not os.path.exists(dist_source):
            print(building_error_msg)
            sys.exit(-1)
        if os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
        copytree(dist_source, TEMP_PATH)
    else:
        print("Do nothing for release installation")


def get_bigdl_packages():
    bigdl_python_home = os.path.abspath(__file__ + "/..")
    bigdl_packages = ['bigdl.share.friesian']
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
    metadata = dict(
        name='bigdl-friesian',
        version=VERSION,
        description='Large-scale End-to-End Recommendation Solution on Big Data',
        author='BigDL Authors',
        author_email='bigdl-user-group@googlegroups.com',
        license='Apache License, Version 2.0',
        url='https://github.com/intel-analytics/analytics-zoo',
        packages=get_bigdl_packages(),
        # TODO: add bigdl-orca as dependency and change to bigdl-orca[ray] in extras_require
        install_requires=['pyspark==2.4.6', 'conda-pack==0.3.1',
                          'packaging', 'filelock'],
        extras_require={'train': ['ray==1.2.0', 'psutil', 'aiohttp==3.7.0', 'aioredis==1.1.0',
                                'setproctitle', 'hiredis==1.1.0', 'async-timeout==3.0.1']},
        dependency_links=['https://d3kbcqa49mib13.cloudfront.net/spark-2.0.0-bin-hadoop2.7.tgz'],
        include_package_data=True,
        package_data={"bigdl.share.friesian": ['lib/bigdl-friesian*.jar']},
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
        init_env()
        setup_package()
    except Exception as e:
        raise e
    finally:
        if build_from_source() and os.path.exists(TEMP_PATH):
            rmtree(TEMP_PATH)
