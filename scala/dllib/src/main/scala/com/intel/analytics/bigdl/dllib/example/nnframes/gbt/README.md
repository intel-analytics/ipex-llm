# Prepare

## Environment
- Spark 2.4 or Spark 3.1
- BigDL 2.0 

## Data Prepare

### BigDL nightly build

You can download [here](https://bigdl.readthedocs.io/en/latest/doc/release.html).
For spark 2.4 you need `bigdl-dllib-spark_2.4.6-0.14.0-build_time-jar-with-dependencies.jar` or `bigdl-dllib-spark_3.1.2-0.14.0-build_time-jar-with-dependencies.jar` for spark 3.1 . 

# GBT On Criteo-click-logs-dataset
## Download data
You can download the criteo-1tb-click-logs-dataset from [here](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/). Then unzip the files you downloaded and Split 1g data to a folder.

## Train
```
spark-submit \
  --master local[4] \
  --conf spark.task.cpus=4 \
  --class com.intel.analytics.bigdl.dllib.example.nnframes.gbt.gbtClassifierTrainingExampleOnCriteoClickLogsDataset \
  --num-executors 2 \
  --executor-cores 4 \
  --executor-memory 4G \
  --driver-memory 10G \
  /path/to/bigdl-dllib-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar \
  -i /path/to/preprocessed-data/saved -s /path/to/model/saved -I max_Iter -d max_depth
```

parameters:
- input_path: String. Path to criteo-click-logs-dataset.
- modelsave_path: String. Path to model to be saved.
- max_iter: Int. Training max iter.
- max_depth: Int. Tree max depth.

The tree of folder `/path/to/model/saved` is:
```
/path/to/model/saved
├── data
└── metadata
    ├── part-00000
    └── _SUCCESS
```
