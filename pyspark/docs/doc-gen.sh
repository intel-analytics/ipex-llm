#!/bin/bash

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


isSphinxInstalled=$(pip show Sphinx &>/dev/null; echo $?)
if [ ! $isSphinxInstalled -eq 0 ]; then
 pip install -U Sphinx
fi

isSphinxInstalled=$(pip show Sphinx &>/dev/null; echo $?)
if [ ! $isSphinxInstalled -eq 0 ]; then
 echo "Please install Sphinx"
 exit 1
fi

isPy4jInstalled=$(pip show Py4j &>/dev/null; echo $?)
if [ ! $isPy4jInstalled -eq 0 ]; then
 pip install -U Py4j
fi

isPy4jInstalled=$(pip show Py4j &>/dev/null; echo $?)
if [ ! $isPy4jInstalled -eq 0 ]; then
 echo "Please install Py4j"
 exit 1
fi

DOCS_DIR="$( cd "$( dirname "$0" )" && pwd)"

sphinx-apidoc -F -f -a -H BigDL -A Intel -o ./ ../ ${DOCS_DIR}/../test/* ${DOCS_DIR}/../setup.py

if [ ! $SPARK_HOME ] || [ -z $SPARK_HOME ]; then
 echo 'Cannot find SPARK_HOME . Please set SPARK_HOME first.'
 exit 1
fi

PYSPARK=$(find -L $SPARK_HOME -name pyspark.zip)
if [ -z $PYSPARK ]; then
 echo 'Cannot find pyspark.zip. Please set SPARK_HOME correctly'
 exit 1
fi

sed -i "/sys.path.insert(0/i sys.path.insert(0, '.')\nsys.path.insert(0, u'$PYSPARK')" conf.py
sed -i "/^extensions/s/^extensions *=/extensions +=/" conf.py
sed -i "/^extensions/i extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'bigdl_pytext']" conf.py
sed -i "/^html_theme/c html_theme = 'sphinxdoc'" conf.py

#remove sidebar 
#sed -i -e '108d;109d;110d;111d;112d;113d;114d;115d;116d' conf.py

make clean; make html;
