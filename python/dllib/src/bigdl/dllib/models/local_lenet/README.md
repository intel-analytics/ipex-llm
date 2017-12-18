# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detailed information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

This example would show how to train and inference a LeNet model in pure local mode without using
Spark local or Spark distributed cluster.
 
## How to run this example:

```
pip install BigDL
```

```
export SPARK_DRIVER_MEMORY=2g
python ${BigDL_HOME}/pyspark/bigdl/models/local_lenet/local_lenet.py
```