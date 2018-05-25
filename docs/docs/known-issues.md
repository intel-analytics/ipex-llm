* If you encounter the following exception when calling the Python API of Analytics Zoo:
```
Py4JJavaError: An error occurred while calling z:org.apache.spark.bigdl.api.python.BigDLSerDe.loads.
: net.razorvine.pickle.PickleException: expected zero arguments for construction of ClassDict (for numpy.dtype)
```
you may need to check whether your input argument involves Numpy types (such as `numpy.int64`). See [here](https://issues.apache.org/jira/browse/SPARK-12157) for the related issue.

For example, invoking `np.min`, `np.max`, `np.unique`, etc. will return type `numpy.int64`. One way to solve this is to use `int()` to convert a number of type `numpy.int64` to a Python int.
