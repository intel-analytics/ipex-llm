# LeNet5 Model on MNIST

LeNet5 is a classical CNN model used in digital number classification. For detailed information,
please refer to <http://yann.lecun.com/exdb/lenet/>.

The model used here is exactly the same as the model in [../lenet/lenet5.py](../lenet/lenet5.py).

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
* ```--batchSize``` an option that can be used to set batch size.
* ```--max_epoch``` an option that can be used to set how many epochs for which the model is to be trained.
* ```--dataPath``` an option that can be used to set the path for downloading mnist data, the default value is /tmp/mnist. Make sure that you have write permission to the specified path.