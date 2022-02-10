# Prepare

## Environment
- Spark 2.4 or Spark 3.1
- BigDL 2.0 

## Data Prepare

### BigDL nightly build

You can download [here](https://bigdl.readthedocs.io/en/latest/doc/release.html).
For spark 2.4 you need `bigdl-dllib-spark_2.4.6-0.14.0-build_time-jar-with-dependencies.jar` or `bigdl-dllib-spark_3.1.2-0.14.0-build_time-jar-with-dependencies.jar` for spark 3.1 . 


### UCI iris.data

You can download iris.data [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

# XGBoostClassifier Train Example
## Run:

command:
```
spark-submit \
  --master local[4] \
  --conf spark.task.cpus=2 \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExample \
  /path/to/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  /path/to/iris.data 2 100 /path/to/model_to_be_saved
```

You will get output like:
```
[INFO] [12/08/2021 16:05:41.989] [RabitTracker-akka.actor.default-dispatcher-22] [akka://RabitTracker/user/Handler] [591298]    train-merror:0.000000   eval1-merror:0.000000     eval2-merror:0.125000
```
And tree of folder `/path/to/model_to_be_saved` :
```
.
├── data
│   └── XGBoostClassificationModel
└── metadata
    ├── part-00000
    └── _SUCCESS
```
parameters:
- path_to_iris.data : String
- num_threads : Int
- num_round : Int 
- path_to_model_to_be_saved : String

**note: make sure num_threads is larger than spark.task.cpus.**

# XGBoostClassifier Predict Example
## Run:
```
spark-submit \
  --master local[4] \
  --conf spark.task.cpus=2 \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierPredictExample \
  /path/to/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  /path/to/iris.data /path/to/model/saved
```
You will get output like:
```
+------------+-----------+------------+-----------+-----------+--------------------+--------------------+----------+
|sepal length|sepal width|petal length|petal width|      class|       rawPrediction|         probability|prediction|
+------------+-----------+------------+-----------+-----------+--------------------+--------------------+----------+
|         5.1|        3.5|         1.4|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.9|        3.0|         1.4|        0.2|Iris-setosa|[2.94163084030151...|[0.98863482475280...|       0.0|
|         4.7|        3.2|         1.3|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.6|        3.1|         1.5|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.0|        3.6|         1.4|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.4|        3.9|         1.7|        0.4|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.6|        3.4|         1.4|        0.3|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.0|        3.4|         1.5|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.4|        2.9|         1.4|        0.2|Iris-setosa|[2.94163084030151...|[0.97911602258682...|       0.0|
|         4.9|        3.1|         1.5|        0.1|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.4|        3.7|         1.5|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.8|        3.4|         1.6|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         4.8|        3.0|         1.4|        0.1|Iris-setosa|[2.94163084030151...|[0.98863482475280...|       0.0|
|         4.3|        3.0|         1.1|        0.1|Iris-setosa|[2.94163084030151...|[0.98863482475280...|       0.0|
|         5.8|        4.0|         1.2|        0.2|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.7|        4.4|         1.5|        0.4|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.4|        3.9|         1.3|        0.4|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.1|        3.5|         1.4|        0.3|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.7|        3.8|         1.7|        0.3|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
|         5.1|        3.8|         1.5|        0.3|Iris-setosa|[2.94163084030151...|[0.99256813526153...|       0.0|
+------------+-----------+------------+-----------+-----------+--------------------+--------------------+----------+
only showing top 20 rows
```
parameters:
- path_to_iris.data : String
- path_to_model_saved : String

# XGBoost On Criteo-click-logs-dataset
## Download data
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Then unzip the files you downloaded to a folder.

## Train
```
spark-submit \
  --master local[16] \
  --conf spark.task.cpus=16 \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExampleOnCriteoClickLogsDataset \
  --conf spark.scheduler.maxRegisteredResourcesWaitingTime=50000000 \
  --conf spark.worker.timeout=60000000 \
  --conf spark.network.timeout=10000000 \
  --conf spark.starvation.timeout=2500000 \
  --conf spark.speculation=false \
  --conf spark.executor.heartbeatInterval=10000000 \
  --conf spark.sql.shuffle.partitions=200 \
  --conf spark.shuffle.io.maxRetries=8 \
  --num-executors 16 \
  --executor-cores 4 \
  --executor-memory 4G \
  --driver-memory 32G \
  /path/to/bigdl-dllib-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  /path/to/preprocessed-data/saved /path/to/model/saved 4
```

parameters:
- input_path: String. Path to criteo-click-logs-dataset.
- modelsave_path: String. Path to model to be saved.
- num_threads: Int. Xgboost train threads.
