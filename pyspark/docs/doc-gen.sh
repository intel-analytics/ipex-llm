#!/bin/bash
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

sphinx-apidoc -F -f -a -H BigDL -A Intel -o ./ ../

if [ ! $SPARK_HOME ] || [ -z $SPARK_HOME ]; then
 echo 'Cannot find SPARK_HOME . Please set SPARK_HOME first.'
 exit 1
fi

PYSPARK=$(find $SPARK_HOME -name pyspark.zip)
if [ -z $PYSPARK ]; then
 echo 'Cannot find pyspark.zip. Please set SPARK_HOME correctly'
 exit 1
fi

sed -i "/sys.path.insert(0/i sys.path.insert(0, '.')\nsys.path.insert(0, u'$PYSPARK')" conf.py
sed -i "/^extensions/s/^extensions *=/extensions +=/" conf.py
sed -i "/^extensions/i extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'bigdl_pytext']" conf.py
sed -i "/^html_theme/c html_theme = 'sphinxdoc'" conf.py

make clean; make html;
