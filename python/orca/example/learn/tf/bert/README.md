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

In above commands
* `--cluster_mode` The mode of spark cluster, supporting local, yarn and spark-submit. Default is "local".

## Results
You can find the logs for training:
```
(Worker pid=627) INFO:tensorflow:loss = 5.5479174, step = 0
(Worker pid=627) INFO:tensorflow:loss = 5.401509, step = 1 (57.550 sec)
(Worker pid=627) INFO:tensorflow:global_step/sec: 0.0173761
```

