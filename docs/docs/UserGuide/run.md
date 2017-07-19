

## **Use an Interactive Shell**
You can try BigDL easily using the Spark interactive shell. Run below command to start spark shell with BigDL support:
```bash
$ SPARK_HOME/bin/spark-shell --properties-file dist/conf/spark-bigdl.conf    \
  --jars bigdl-VERSION-jar-with-dependencies.jar
```
You will see a welcome message looking like below:
```
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 1.6.0
      /_/

Using Scala version 2.10.5 (Java HotSpot(TM) 64-Bit Server VM, Java 1.7.0_79)
Spark context available as sc.
scala> 
```

To use BigDL, you should first initialize the engine as below. 
```scala
scala>import com.intel.analytics.bigdl.utils.Engine
scala>Engine.init
```

Once the engine is successfully initialted, you'll be able to play with BigDL API's. 
For instance, to experiment with the ````Tensor```` APIs in BigDL, you may try below code:
```scala
scala> import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.Tensor

scala> Tensor[Double](2,2).fill(1.0)
res9: com.intel.analytics.bigdl.tensor.Tensor[Double] =
1.0     1.0
1.0     1.0
[com.intel.analytics.bigdl.tensor.DenseTensor of size 2x2]
```

---

## **Run a BigDL Program**
You can run a BigDL program, e.g., the [VGG](https://github.com/intel-analytics/BigDL/tree/master/spark/dl/src/main/scala/com/intel/analytics/bigdl/models/vgg) training, as a standard Spark program (running in either local mode or cluster mode) as follows:

1. Download the CIFAR-10 data from [here](https://www.cs.toronto.edu/%7Ekriz/cifar.html). Remember to choose the binary version.

```
  # Spark local mode
  spark-submit --master local[core_number] --class com.intel.analytics.bigdl.models.vgg.Train \
  dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -f path_to_your_cifar_folder \
  -b batch_size

  # Spark standalone mode
  spark-submit --master spark://... --executor-cores cores_per_executor \
  --total-executor-cores total_cores_for_the_job \
  --class com.intel.analytics.bigdl.models.vgg.Train \
  dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -f path_to_your_cifar_folder \
  -b batch_size

  # Spark yarn mode
  spark-submit --master yarn --deploy-mode client \
  --executor-cores cores_per_executor \
  --num-executors executors_number \
  --class com.intel.analytics.bigdl.models.vgg.Train \
  dist/lib/bigdl-VERSION-jar-with-dependencies.jar \
  -f path_to_your_cifar_folder \
  -b batch_size
```

  The parameters used in the above command are:

  * -f: The folder where your put the CIFAR-10 data set. Note in this example, this is just a local file folder on the Spark driver; as the CIFAR-10 data is somewhat small (about 120MB), we will directly send it from the driver to executors in the example.

  * -b: The mini-batch size. The mini-batch size is expected to be a multiple of *total cores* used in the job. In this example, the mini-batch size is suggested to be set to *total cores * 4*

If you are to run your own program, do remember to create SparkContext and initialize the engine before call other BigDL API's, as shown below. 
```scala
 // Scala code example
 val conf = Engine.createSparkConf()
 val sc = new SparkContext(conf)
 Engine.init
```

