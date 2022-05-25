# Orca Tensorflow Bert Pretraining example with Wiki Dataset

We demonstrate how to easily run synchronous distributed Tensorflow training using Tensorflow Estimator of Project Orca in BigDL. This example runs bert pretraining for Wiki Dataset on a CPU cluster. See [here](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert) for the original GPU/TPU version of this example.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n bigdl python=3.7 #bigdl is conda enviroment name, you can set another name you like.
conda activate bigdl
pip install tensorflow==1.15
pip install --pre --upgrade bigdl-orca
```

## Prepare Dataset
You could download and generate the TFRecords for the Wiki Dataset by refering to the instructions [here](https://github.com/mlcommons/training/tree/master/language_model/tensorflow/bert#download-and-preprocess-datasets).


## Run example
You can run this example on local mode and yarn client mode.

- Run with Spark Local mode:
```bash
python run_pretraining.py \
  --bert_config_file=<path to bert_config.json> \
  --output_dir=/tmp/output/ \
  --input_file="<tfrecord dir>/part*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=./checkpoint/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24 \
  --cluster_mode=local
```

- Run with Yarn Client mode:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server:
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)
export HADOOP_HDFS_HOME=$HADOOP_HOME

python run_pretraining.py \
  --bert_config_file=<path to bert_config.json> \
  --output_dir=/tmp/output/ \
  --input_file="hdfs:///<tfrecord dir>/part*" \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=ckpt/tf1_ckpt/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=107538 \
  --num_warmup_steps=1562 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --num_gpus=8 \
  --train_batch_size=24 \
  --num_executors=10 \
  --cluster_mode=yarn
```

- Run with spark-submit mode:

The args for python is similar as `local` mode and `yarn` mode. For how to run k8s cluster mode on
BigDL, you could refer to [BigDL k8s user guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/k8s.html#k8s-cluster-mode). 

You could also follow below steps to run on kubernetes in cluster mode with NFS.
1. The prepared dataset should be saved on NFS.
2. Create an model output path on NFS to save checkpoints.
3. Copy the bert example [repo](../bert) to NFS. Zip all files in the example repo, saved as "bert.zip"
4. Follow the step 1-3 in [BigDL k8s user guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/k8s.html#k8s-cluster-mode) to set up k8s for BigDL.
5. Refer to below commands to submit your BigDL program.

```bash
export MOUNT_PATH="/bigdl2.0/data"
export BERT_CONFIG_PATH="${MOUNT_PATH}/<your data path>/bert_config.json"
export INPUT_FILE="${MOUNT_PATH}/<your data path>/tfrecord/part*"
export CHECKPOINT_PATH="${MOUNT_PATH}/<your data path>/tf1_ckpt"
export OUTPUT_DIR="${MOUNT_PATH}/<your output dir>"

/opt/spark/bin/spark-submit \
--master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
--deploy-mode cluster \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--name bigdl-test \
--conf spark.kubernetes.container.image="intelanalytics/bigdl-k8s:latest" \
--conf spark.executor.instances=8 \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=$MOUNT_PATH \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=$MOUNT_PATH \
--conf spark.kubernetes.driverEnv.http_proxy=http://child-prc.intel.com:913 \
--conf spark.kubernetes.driverEnv.https_proxy=http://child-prc.intel.com:913 \
--conf spark.kubernetes.executorEnv.http_proxy=http://child-prc.intel.com:913 \
--conf spark.kubernetes.executorEnv.https_proxy=http://child-prc.intel.com:913 \
--conf spark.kubernetes.container.image.pullPolicy=Always \
--archives file://$MOUNT_PATH/env.tar.gz#python_env \
--conf spark.pyspark.driver.python=python_env/bin/python \
--conf spark.pyspark.python=python_env/bin/python \
--conf spark.executorEnv.PYTHONHOME=python_env \
--conf spark.kubernetes.executor.deleteOnTermination=false \
--conf spark.kubernetes.file.upload.path=$MOUNT_PATH/upload/ \
--conf spark.kubernetes.driver.podTemplateFile=$MOUNT_PATH/spark-driver-template.yaml \
--conf spark.kubernetes.executor.podTemplateFile=$MOUNT_PATH/spark-driver-template.yaml \
--executor-cores 44 \
--executor-memory 100g \
--total-executor-cores 352\
--driver-cores 4 \
--driver-memory 100g \
--properties-file /opt/bigdl-2.1.0-SNAPSHOT/conf/spark-bigdl.conf \
--py-files local://$MOUNT_PATH/bert.zip,local://$MOUNT_PATH/bert/run_pretraining.py \
--conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
--conf spark.sql.catalogImplementation='in-memory'  \
--conf spark.driver.extraClassPath=local:///opt/bigdl-2.1.0-SNAPSHOT/jars/* \
--conf spark.executor.extraClassPath=local:///opt/bigdl-2.1.0-SNAPSHOT/jars/* \
local://${MOUNT_PATH}/bert/run_pretraining.py \
  --bert_config_file=$BERT_CONFIG_PATH \
  --output_dir=$OUTPUT_DIR \
  --input_file=$INPUT_FILE \
  --nodo_eval \
  --do_train \
  --eval_batch_size=8 \
  --learning_rate=0.0001 \
  --init_checkpoint=$CHECKPOINT_PATH/model.ckpt-28252 \
  --iterations_per_loop=1000 \
  --max_predictions_per_seq=76 \
  --max_seq_length=512 \
  --num_train_steps=10 \
  --num_warmup_steps=5 \
  --optimizer=lamb \
  --save_checkpoints_steps=6250 \
  --start_warmup_step=0 \
  --train_batch_size=64 \
  --max_eval_steps=2 \
  --cluster_mode=spark-submit

```

A sample `spark-driver-template.yaml` is as below.
```
apiVersion: v1
kind: Pod
spec:
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 40Gi
  containers:
  - image: "intelanalytics/bigdl-k8s:latest"
    volumeMounts:
      - mountPath: /dev/shm
        name: dshm
```

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local, yarn and spark-submit. Default is "local".

## Results
You can find the logs for training:
```
(Worker pid=627) INFO:tensorflow:loss = 5.5479174, step = 0
(Worker pid=627) INFO:tensorflow:loss = 5.401509, step = 1 (57.550 sec)
(Worker pid=627) INFO:tensorflow:global_step/sec: 0.0173761
```

