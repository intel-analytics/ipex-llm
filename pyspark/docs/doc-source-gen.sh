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

sphinx-apidoc -F -f -a -H BigDL -A Intel -o ./ ../dl

PYSPARK=$SPARK_HOME'/python/lib/pyspark.zip'
if [ ! -f "$PYSPARK" ]; then
  echo "pyspark.zip for spark is not found."
  echo "Please input the path of spark pyspark.zip."
  echo "For example: spark-1.6.0-bin-hadoop2.6/python/lib/pyspark.zip"
  read PYSPARK
fi
sed -i "/sys.path.insert(0/i sys.path.insert(0, '.')\nsys.path.insert(0, u'$PYSPARK')" conf.py
sed -i "/^extensions/s/^extensions *=/extensions +=/" conf.py
sed -i "/^extensions/i extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'bigdl_pytext']" conf.py
sed -i "/^html_theme/c html_theme = 'sphinxdoc'" conf.py
