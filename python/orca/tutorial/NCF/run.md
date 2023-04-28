# Orca NCF Tutorial

## Download Dataset
- https://files.grouplens.org/datasets/movielens/ml-1m.zip
- https://files.grouplens.org/datasets/movielens/ml-100k.zip

```bash
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ./ml-1m.zip

# Upload the dataset to HDFS if you run on yarn:
hdfs dfs -mkdir -p hdfs://ip:port/data/NCF
hdfs dfs -put ml-1m hdfs://ip:port/data/NCF
hdfs dfs -chmod -R 777 hdfs://ip:port/data/NCF

# Upload the dataset to NFS if you run on k8s:
cp -r ml-1m /bigdl/nfsdata
```
 
## Arguments
+ `--data_dir`: The path to load data from local or remote resources (default to be `./`). You should input an HDFS path if you run on yarn.
+ `--dataset`: The name of the dataset, ml-1m or ml-100k (default to be `ml-1m`).
+ `--model_dir`: The path to save model and logs (default to be `./`). You should input an HDFS path if you run on yarn.
+ `--cluster_mode`: The cluster mode, one of `local`, `yarn-client`, `yarn-cluster`, `k8s-client`, `k8s-cluster`, `spark-submit` or `bigdl-submit` (default to be `local`).
+ `--backend`: The backend of Orca Estimator, either ray or spark (default to be `spark`).
+ `--workers_per_node`: The number of workers on each node (default to be `1`).
+ `--tensorboard`: Whether to use TensorBoard as the train callback.
+ `--lr_scheduler`: Whether to use learning rate scheduler for training.

## Prepare the environment
```bash
conda create -n NCF-orca python=3.7
conda activate NCF-orca

# spark backend
pip install --pre --upgrade bigdl-orca-spark3
# ray backend
pip install --pre --upgrade bigdl-orca-spark3[ray]
# TensorFlow
pip install tensorflow==2.9.0
# PyTorch
pip install torch torchvision torchmetrics==0.10.0 tqdm
# PyTorch DataLoader
pip install pandas scikit-learn
# XShards
pip install pandas scikit-learn pyarrow
# Use Tensorboard Callback
pip install tensorboard
```

## 1. Run on local

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir file:///local/path/to/NCF
```
You can replace the file name with [other files](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/NCF/README.md). Need to run the train script first before resume training or prediction.

### Results

Check the processed data and saved model files after running the train script:
+ `NCF_model`: The trained model 
+ `config.json`: The model configuration for predict and resume training
+ Processed dataframe parquet/xshards under `data_dir`
<details>
<summary> Click to see the console output </summary>

```bash
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
```

</details>

## 2.1 Run on YARN with python command

### Run Command
```bash
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF --cluster_mode yarn-client
python pytorch_train_spark_dataframe.py --data_dir hdfs://ip:port/data/NCF --cluster_mode yarn-cluster
```

## 2.2 Run on YARN with bigdl-submit
```bash
conda pack -o environment.tar.gz
```

### Run Command
+ For `yarn-client` mode
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
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=environment/bin/python \
    pytorch_train_spark_dataframe.py \
    --cluster_mode bigdl-submit \
    --data_dir hdfs://ip:port/data/NCF
```

+ For `yarn-cluster` mode
```bash
bigdl-submit \
   --master yarn \
   --deploy-mode cluster \
   --num-executors 2 \
   --executor-cores 4 \
   --executor-memory 10g \
   --driver-cores 2 \
   --driver-memory 2g \
   --py-files process_spark_dataframe.py,pytorch_model.py,utils.py \
   --archives environment.tar.gz#environment \
   --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
   --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
   pytorch_train_spark_dataframe.py \
   --cluster_mode bigdl-submit \
   --data_dir hdfs://ip:port/data/NCF
```

## 2.3 Run on YARN with spark-submit

### Prepare the Environment
- **Do not** install bigdl-orca-spark3 in the conda environment.
- Install the dependencies of bigdl-orca as listed in the dependency files.
- `conda pack -o environment.tar.gz`
- Download Spark and set `SPARK_HOME` and `SPARK_VERSION`.
- Download BigDL and set `BIGDL_HOME` and `BIGDL_VERSION`.

### Run Command
+ For `yarn-client` mode
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
   --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,pytorch_model.py \
   --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
   pytorch_train_spark_dataframe.py \
   --cluster_mode spark-submit \
   --data_dir hdfs://ip:port/data/NCF
```

+ For `yarn-cluster` mode
```bash
${SPARK_HOME}/bin/spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 2 \
    --executor-cores 4 \
    --executor-memory 10g \
    --driver-memory 2g \
    --archives environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=environment/bin/python \
    --conf spark.executorEnv.PYSPARK_PYTHON=environment/bin/python \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,process_spark_dataframe.py,pytorch_model.py,utils.py \
    --jars ${BIGDL_HOME}/jars/bigdl-assembly-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar \
    pytorch_train_spark_dataframe.py  --cluster_mode spark-submit --data_dir hdfs://ip:port/data/NCF
```

## 3.1 Run on K8s with python command
You need to run the python command inside a docker container.

### Prepare the Environment
Pull the BigDL bigdl-k8s image
```bash
# For the release version, e.g. 2.3.0
sudo docker pull intelanalytics/bigdl-k8s:version

# For the latest nightly build version
sudo docker pull intelanalytics/bigdl-k8s:latest
```

Create the client container
```bash
export RUNTIME_DRIVER_HOST=$( hostname -I | awk '{print $1}' )

sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v /path/to/nfsdata:/bigdl/nfsdata \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/bigdl-k8s:version \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=nfsvolumeclaim \
    -e RUNTIME_DRIVER_HOST=${RUNTIME_DRIVER_HOST} \
    intelanalytics/bigdl-k8s:version bash
sudo docker exec -it <containerID> bash
```

### Run Command
For `k8s-client` mode
```bash
python pytorch_train_spark_dataframe.py --data_dir /bigdl/nfsdata/ --cluster_mode k8s-client
```

For `k8s-cluster` mode 
```bash
conda pack -o environment.tar.gz
cp environment.tar.gz /bigdl/nfsdata
cp *.py /bigdl/nfsdata
python /bigdl/nfsdata/pytorch_train_sparkdataframe.py --data_dir /bigdl/nfsdata --cluster_mode k8s-cluster
```

## 3.2 Run on K8s with spark submit
You need to run the python command inside a docker container.
+ Do not install bigdl-orca-spark3 in the conda environment.
+ Install the dependencies of bigdl-orca as listed in the dependency files.

### Run Command
+ For `k8s-client` mode
```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode client \
    --name orca-k8s-client-tutorial \
    --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --num-executors 2 \
    --executor-cores 4 \
    --total-executor-cores 8 \
    --executor-memory 10g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=environment/bin/python \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,pytorch_model.py \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    pytorch_train_sparkdataframe.py --cluster_mode spark-submit --data_dir /bigdl/nfsdata/
```

+ For `k8s-cluster` mode

```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode cluster \
    --name orca-k8s-cluster-tutorial \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
    --num-executors 2 \
    --executor-cores 4 \
    --total-executor-cores 8 \
    --executor-memory 10g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives /bigdl/nfsdata/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=environment/bin/python \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/bigdl/nfsdata/process_spark_dataframe.py,/bigdl/nfsdata/pytorch_model.py,/bigdl/nfsdata/utils.py \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    /bigdl/nfsdata/pytorch_train_sparkdataframe.py --cluster_mode spark-submit --data_dir /bigdl/nfsdata/
```
