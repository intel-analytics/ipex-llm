This page lists solutions to some common questions.

1. __ImportError__: from zoo.pipeline.api.keras.layers import *
    - Check if the path is pointing to python-api.zip: ```--py-files ${ANALYTICS_ZOO_PY_ZIP} ```
    - Check if the path is pointing to python-api.zip:
    
    ``` export PYTHONPATH=${ANALYTICS_ZOO_PY_ZIP}:$PYTHONPATH```

2. Python in worker has a different version than that in driver
    - ```export PYSPARK_PYTHON=/usr/local/bin/python3.5```  This path should be valid on every worker node.
    - ```export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.5```  This path should be valid on every driver node.
  
3. __TypeError__: 'JavaPackage' object is not callable
    - Check if every path within the launch script is valid especially the path that ends with jar.
    - If there are extra jars involved, check if the Spark version Analytics Zoo is built and the Spark version the extra jar is built are compatible.

4. java.lang.__NoSuchMethodError__:XXX or __Py4JError__: ofFloat does not exist in the JVM
    - Check if the Spark version matches, i.e check if you are using Spark 2.x but the underneath Analytics Zoo is compiled with Spark 1.6.
    - If there are extra jars involved, also check if the Spark version matches.
