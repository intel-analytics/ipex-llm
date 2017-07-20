
* [Install via pip](install-from-pip.md)

## **Use an Interactive Shell**
 * type `python` in commandline to start a REPL
 * [Example code to verify if run successfully](#code.verification)

## **Use Jupyter Notebook**

 * Start jupyter notebook as you normally did, e.g.
 ```bash
 jupyter notebook --notebook-dir=./ --ip=* --no-browser
 ```
 * [Example code to verify if run successfully](#code.verification)



<a name="code.verification"></a>
## Code ##
```python
 from bigdl.util.common import *
 from pyspark import SparkContext
 from bigdl.nn.layer import *
 import bigdl.version
 
 sc = SparkContext.getOrCreate(conf=create_spark_conf()) # create sparkcontext with bigdl configuration
 init_engine() # prepare the bigdl environment 
 bigdl.version.__version__ # Get the current BigDL version
 linear = Linear(2, 3) # Try to create a Linear layer
 
```


