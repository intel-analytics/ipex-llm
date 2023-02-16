# Orca NCF Tutorial

## Download Dataset
- https://files.grouplens.org/datasets/movielens/ml-1m.zip
- https://files.grouplens.org/datasets/movielens/ml-100k.zip

```bash
hdfs dfs -mkdir -p hdfs://ip:port/data/NCF
hdfs dfs -put ml-1m hdfs://ip:port/data/NCF
```

## Arguments
TODO

## Prepare the environment
```bash
conda create -n NCF-yarn python=3.7
conda activate NCF-yarn

# spark backend
pip install --pre --upgrade bigdl-orca-spark3
# ray backend
pip install --pre --upgrade bigdl-orca-spark3[ray]
# TensorFlow
pip install tensorflow==2.9.0
# PyTorch
pip install torch torchvision torchmetrics==0.10.0 tqdm
# XShards
pip install pandas scikit-learn pyarrow
# Tensorboard
pip install tensorboard
```

## 1. Run on local

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir file:///local/path/to/NCF/ml-1m
```
You can replace the file name with other files. Need to run the train script first before resume training or prediction.

### Results

The above command will return the following files
+ `NCF_model`: A zip file containing the trained PyTorch model 
+ `config.json`: The model configuration for predict and resume training
+ Processed dataframe parquet under `data_dir`
    + `ml-1m/train_processed_dataframe.parquet`
    + `ml-1m/test_processed_dataframe.parquet`
<details>
<summary> Click to see the console output </summary>

```bash
Loading data...
Train results:
num_samples: 2882458
epoch: 1.0
batch_count: 282.0
train_loss: 0.3417230605540067
last_train_loss: 0.29314390341794283
val_accuracy: 0.874625027179718
val_precision: 0.7700153589248657
val_recall: 0.5436215400695801
val_loss: 0.2966938563155457
val_num_samples: 720335.0

num_samples: 2882458
epoch: 2.0
batch_count: 282.0
train_loss: 0.2704732511598162
last_train_loss: 0.26464769959350676
val_accuracy: 0.8847938776016235
val_precision: 0.7885202169418335
val_recall: 0.5895587801933289
val_loss: 0.2665152659829276
val_num_samples: 720335.0

Evaluation results:
num_samples: 720340
Accuracy: 0.8834050297737122
Precision: 0.7855872511863708
Recall: 0.5875657200813293
val_loss: 0.2684867502365162
```

</details>

## 2-1 Run on YARN with python command

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF/ml-1m  --cluster_mode yarn-client
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF/ml-1m  --cluster_mode yarn-cluster
```

## 2-2 Run on YARN with bigdl-submit
```bash
conda pack -o environment.tar.gz
```

### Run Command
```bash
bigdl-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 10g \
    --driver-memory 2g \
    --py-files process_spark_dataframe.py,pytorch_model.py,utils.py \
    --archives environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=environment/bin/python \
    pytorch_train_spark_dataframe.py \
    --cluster_mode bigdl-submit \
    --data_dir hdfs://ip:port/data/NCF/ml-1m
```

## 2-3 Run on YARN with spark-submit

### Prepare the Environment
- Do not install bigdl-orca-spark3 in the conda environment.
- Install the dependencies of bigdl-orca as listed in the dependency files.
- `conda pack -o environment.tar.gz`
- Download Spark and set SPARK_HOME and SPARK_VERSION.
- Download BigDL and set BIGDL_HOME and BIGDL_VERSION.

### Run Command
```bash
${SPARK_HOME}/bin/spark-submit \
   --master yarn \
   --deploy-mode client \
   --num-executors 2 \
   --executor-cores 4 \
   --executor-memory 10g \
   --driver-memory 2g \
   --archives environment.tar.gz#environment \
   --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
   --conf spark.pyspark.driver.python=python \
   --conf spark.pyspark.python=environment/bin/python \
   --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,process_spark_dataframe.py,pytorch_model.py,utils.py \
   --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
   pytorch_train_spark_dataframe.py \
   --cluster_mode spark-submit \
   --data_dir hdfs://ip:port/data/NCF/ml-1m
```
