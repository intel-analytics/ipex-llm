# Orca NCF Tutorial

## Download Dataset
- https://files.grouplens.org/datasets/movielens/ml-1m.zip
- https://files.grouplens.org/datasets/movielens/ml-100k.zip

```bash
hdfs dfs -mkdir -p hdfs://ip:port/data/NCF
hdfs dfs -put ml-1m hdfs://ip:port/data/NCF
```

## Arguments
+ `--data_dir`: The path to load data from local or remote resources (default to be `./ml-1m`).
+ `--model_dir`: The path to save model and logs (default to be `./`).
+ `--cluster_mode`: The cluster mode, such as `local`, `yarn-client`, `yarn-cluster`, `k8s-client`, `k8s-cluster`, `spark-submit` or `bigdl-submit` (default to be `local`).
+ `--backend`: The backend of Orca Estimator, either ray or spark (default to be `spark`).
+ `--workers_per_node`: The number of workers on each node (default to be `1`).
+ `--tensorboard`: Only valid in train mode, whether to use TensorBoard as the train callback.
+ `--lr_scheduler`: Only valid in train mode, whether to use learning rate scheduler for training.

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
num_samples: 3938789
epoch: 1.0
batch_count: 385.0
train_loss: 0.332098191390063
last_train_loss: 0.2929740669144447
val_accuracy: 0.8761903047561646
val_precision: 0.7543439269065857
val_recall: 0.5766154527664185
val_loss: 0.29692074001052177
val_num_samples: 983776.0

num_samples: 3938789
epoch: 2.0
batch_count: 385.0
train_loss: 0.26862238181653536
last_train_loss: 0.26927183309628583
val_accuracy: 0.8856792449951172
val_precision: 0.7758791446685791
val_recall: 0.6126476526260376
val_loss: 0.2679317132346567
val_num_samples: 983776.0

Evaluation results:                                                             
num_samples: 983858
Accuracy: 0.8859977722167969
Precision: 0.7784695625305176
Recall: 0.6156555414199829
val_loss: 0.26760029486236536
Stopping orca context  
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
