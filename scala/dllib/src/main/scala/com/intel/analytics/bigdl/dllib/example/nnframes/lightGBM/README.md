# Prepare

## Environment
- Spark 3.1
- BigDL 2.2 

## Data Prepare

### BigDL nightly build

You can download [here](https://bigdl.readthedocs.io/en/latest/doc/release.html).
For spark 3.1.2 you need  `bigdl-dllib-spark_3.1.2-2.2.0-build_time-jar-with-dependencies.jar`.  


### UCI iris.data

You can download iris.data [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

# LgbmClassifier Train Example

## Run:

command:
```
spark-submit \
  --master local[4] \
  --conf spark.task.cpus=4 \
  --class com.intel.analytics.bigdl.dllib.example.nnframes.lightGBM.LgbmClassifierTrain \
  --input /path/to/bigdl-dllib-spark_3.1.2-2.2.0-SNAPSHOT-jar-with-dependencies.jar \
  --numIterations 100 \
  --modelSavePath /tmp/lgbm/scala/classifier \
  --partition 4

```