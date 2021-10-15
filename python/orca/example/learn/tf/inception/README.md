# Inception Model on Imagenet
This example demonstrates how to use Analytics-zoo to train a TensorFlow [Inception v1](https://arxiv.org/abs/1409.4842) model on the [ImageNet](http://image-net.org/index) data.

## Environment

We recommend conda to set up your environment. You can install a conda distribution from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
if you haven't already.

```bash
conda create -n analytics-zoo python==3.7
conda activate analytics-zoo
pip install tensorflow==1.15.0
```

Then download and install latest nightly-build Analytics Zoo.

```bash
pip install --pre --upgrade analytics-zoo
```

## Prepare the data

Download raw ImageNet data from http://image-net.org/download-images, extract and provide in this format:
```bash
mkdir train
mv ILSVRC2012_img_train.tar train/
cd train
echo "Extracting training images"
tar -xvf ILSVRC2012_img_train.tar
rm ILSVRC2012_img_train.tar
find . -name "*.tar" | while read CLASS_NAME ; do mkdir -p "${CLASS_NAME%.tar}"; tar -xvf "${CLASS_NAME}" -C "${CLASS_NAME%.tar}"; done
rm *.tar

mkdir val
mv  ILSVRC2012_img_val.tar val/
cd val
echo "Extracting validation images"
tar -xvf ILSVRC2012_img_val.tar
rm ILSVRC2012_img_val.tar
```

## Train the Model

```bash
python inception.py --folder $(raw_imagenet} --imagenet ${imagenet_tfrecords} --cluster_mode yarn --worker_num 4 --cores 54 --memory 175G --batchSize 1792 --maxIteration 62000 --maxEpoch 100 --learningRate 0.0896 --checkpoint /tmp/models/inception
```

In the above commands
* -f: raw ImageNet data path
* --imagenet: ImageNet TFRecords data path. When using HDFS path, you need set environment variables JAVA_HOME,
HADOOP_HDFS_HOME(the location of your HDFS installation), and LD_LIBRARY_PATH(include path to libhdfs.so).
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as optimMethod.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* --batchSize: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number *
core_number. In this example, node_number is 1 and the mini-batch size is suggested to be set to core_number * 4
* --learningRate: inital learning rate. Note in this example, we use a Poly learning rate decay
policy.
* --weightDecay: weight decay.
* --checkpointIteration: the checkpoint interval in iteration.
* --maxLr: optional. Max learning rate after warm up. It has to be set together with warmupEpoch.
* --warmupEpoch: optional. Epoch numbers need to take to increase learning rate from learningRate to maxLR.
* --maxIteration: max iteration
