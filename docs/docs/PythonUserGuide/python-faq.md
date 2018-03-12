This page lists solutions to some common questions.

1. __ImportError__: from bigdl.nn.layer import *
    - Check if the path is pointing to python-api.zip: ```--py-files ${PYTHON_API_ZIP_PATH} ```
    - Check if the path is pointing to python-api.zip: ``` export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH```

2. Python in worker has a different version 2.7 than that in driver 3.5
    - ```export PYSPARK_PYTHON=/usr/local/bin/python3.5```  This path should be valid on every worker node.
    - ```export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.5```  This path should be valid on every driver node.
  
3. __TypeError__: 'JavaPackage' object is not callable
    - Check if every path within the launch script is valid especially the path that ends with jar.
    - If there are extra jars involved, check if the Spark version BigDL is built and the Spark version the extra jar is built are compatible.

4. java.lang.__NoSuchMethodError__:XXX or __Py4JError__: ofFloat does not exist in the JVM
    - Check if the Spark version matches, i.e check if you are using Spark2.x but the underneath BigDL is compiled with Spark1.6.
    - If there are extra jars involved, also check if the Spark version matches.

5. Logs are not displayed properly during the training process.
    - Call the following API before you train your model to redirect spark logs to file and keep BigDL logs in console only.
```python
from bigdl.util.common import *

# by default redirected to `bigdl.log` under the current workspace
redire_spark_logs(log_path="bigdl.log")
show_bigdl_info_logs()
```