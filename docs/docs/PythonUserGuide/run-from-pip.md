## **Precondition**

* [Install via pip](install-from-pip.md)

## **Use an Interactive Shell**
 * Type `python` in the command line to start a REPL
 * Only __Python 2.7__, __Python 3.5__ and __Python 3.6__ are supported for now.
 * Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1 and 2.2.0. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.


## **Run as a local program**
 * If the type of input data is ndarray instead of RDD or DataFrame, the model would be trained or validated in local mode.
 * Check [LeNet](https://github.com/intel-analytics/BigDL/blob/master/pyspark/bigdl/models/local_lenet/README.md) for more details

```python
# X_train, Y_train, X_test are all ndarray and the first dimension is the sample number.
local_optimizer = Optimizer.create(
    model=model_definition,
    training_set=(X_train, Y_train))
local_optimizer.predict(X_test)
local_optimizer.predict_class(X_test)
```

## **Use Jupyter Notebook**
 * Just start jupyter notebook as you normally do, e.g.
```bash
 jupyter notebook --notebook-dir=./ --ip=* --no-browser
```


<a name="code.verification"></a>
## **Example code to verify if BigDL can run successfully**
```python
from bigdl.util.common import *
from pyspark import SparkContext
from bigdl.nn.layer import *
import bigdl.version
 
# create sparkcontext with bigdl configuration
sc = SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[*]"))
init_engine() # prepare the bigdl environment 
bigdl.version.__version__ # Get the current BigDL version
linear = Linear(2, 3) # Try to create a Linear layer
```

## **BigDL Configurations**

* Increase memory
    - export SPARK_DRIVER_MEMORY=20g
* Add extra jars or python packages
    - Call the following API __BEFORE__ creating `SparkContext`:
```python
from bigdl.util.common import Configuration

Configuration.add_extra_jars(jars)
Configuration.add_extra_python_modules(packages)
```
   `jars` can be a string or a list of strings as jar paths to be added.

   `packages` can be a string or a list of strings as python package paths to be appended to `sys.path`.

* Redirect logs
    - If you want to redirect spark logs to file and keep BigDL logs in console only, call the following API before you train your model:
```python
from bigdl.util.common import *

redire_spark_logs(log_path=file path to redirect logs to)
show_bigdl_info_logs()
```
