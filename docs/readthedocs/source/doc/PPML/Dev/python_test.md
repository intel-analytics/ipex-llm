# PPML Python Test Develop Guide

### Write a test
All tests locate in `python/ppml/test`.

#### Single party test
Writing a single party test is just the same as running PPML pipeline, for example, a simple FGBoostRegression pipeline
```python
fl_server = FLServer()
fl_server.build()
fl_server.start()

init_fl_context()
fgboost_regression = FGBoostRegression()
fgboost_regression.fit(...)
```
#### Multiple party test
There are some extra steps for multiple party test, `python/ppml/test/bigdl/ppml/algorithms/test_fgboost_regression.py` could be refered as an example.

Multiple party test requires multiprocessing package. Import the package by
```
import multiprocessing
```
and set the subprocess create config in your class method
```python
class YourTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        multiprocessing.set_start_method('spawn') 
```
And define the function of subprocess
```python
def mock_process(arg1, arg2):
    init_fl_context()
    algo = Algo() # The algorithm to test
    algo.fit(...)
```
and start the process in test method
```python
mock_party1 = Process(target=mock_process, args=(v1, v2))
mock_party1.start()
```

### Debug a test
#### How it works
BigDL uses Spark integrated Py4J to do Python call Java. 

Spark starts the JVM when PySpark code create the SparkContext. This method use a Popen subprocess to call `spark-submit`, which call `spark-class`, and call `java`

#### Set JVM debug mode
First, direct to the `spark-class` file (as there may be multiple class in different packages or copied by python setup during installation) called by PySpark, this could be get by adding a breakpoint after `command = [os.path.join(SPARK_HOME, script)]` in `java_gateway.py` in PySpark lib.

To enable debug, add the JVM args in `spark-class` when call `java`, in the last line `CMD`, change following
```
CMD=("${CMD[@]:0:$LAST}")
```
to
```
CMD=("${CMD[0]}" -agentlib:jdwp=transport=dt_socket,server=y,address=4000,suspend=n "${CMD[@]:1:$LAST}")
```
And in IDEA, create a Run Configuration remote JVM profile. The IDEA will create the VM args automatically.
