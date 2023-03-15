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
pip install pandas scikit-learn
```

## 1. Run on local

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir file:///local/path/to/NCF/ml-1m
```
You can replace the file name with other files. Need to run the train script first before resume training or prediction.

### Results
TODO

## 2-1 Run on YARN with python command

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF  --cluster_mode yarn-client
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF  --cluster_mode yarn-cluster
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
    --py-files pytorch_model.py \
    --archives environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=/home/anaconda3/envs/NCF-yarn/bin/python \
    --conf spark.pyspark.python=environment/bin/python \
    pytorch_train_spark_dataframe.py \
    --cluster_mode bigdl-submit \
    --data_dir hdfs://ip:port/data/NCF
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
   --conf spark.pyspark.driver.python=/home/anaconda3/envs/NCF-yarn/bin/python \
   --conf spark.pyspark.python=environment/bin/python \
   --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,pytorch_model.py \
   --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
   pytorch_train_spark_dataframe.py \
   --cluster_mode spark-submit \
   --data_dir hdfs://ip:port/data/NCF
```
