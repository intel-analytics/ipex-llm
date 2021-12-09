# XGBoostClassifier Train Example

## Environment
- Spark 2.4
- BigDL 2.0 

## Data Prepare

### BigDL nightly build

You can download [here](https://bigdl.readthedocs.io/en/latest/doc/release.html).
You will get jar `bigdl-dllib-spark_2.4.6-0.14.0-build_time-jar-with-dependencies.jar`.

### UCI iris.data

You can download iris.data [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

## Run:

command:
```
spark-submit \
  --master local[2] \
  --conf spark.task.cpus=2  \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost.xgbClassifierTrainingExample \
  /path/to/BigDL/scala/dllib/target/bigdl-dllib-spark_2.4.6-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  /path/to/iris.data 2 100 /path/to/model/saved
```

You will get output like:
```
[INFO] [12/08/2021 16:05:41.989] [RabitTracker-akka.actor.default-dispatcher-22] [akka://RabitTracker/user/Handler] [591298]    train-merror:0.000000   eval1-merror:0.000000     eval2-merror:0.125000
```
And tree of folder `/path/to/model/saved` :
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
- path_to_model_saved : String

note: make sure num_threads is larger than spark.task.cpus.
