# Orca Bigdl Imageinference example

We demonstrate how to easily run synchronous distributed Bigdl training using Bigdl Estimator of Project Orca in Analytics Zoo. 
This example is a demo of image classification: inference with a pre-trained Inception_V1 model based on Spark DataFrame (Dataset). 
See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/nnframes/imageInference) for the original version of this example provided by analytics-zoo.

## Prepare environments
We recommend you to use [Anaconda](https://www.anaconda.com/distribution/#linux) to prepare the environments, especially if you want to run on a yarn cluster(yarn-client mode only).
```
conda create -n zoo python=3.7 #zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install --pre --upgrade analytics-zoo
```

## Prepare Dataset
1. Get the pre-trained Inception-V1 model
Download the pre-trained Inception-V1 model from [Analytics Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model),
and put it in `/tmp/zoo` or other path.

2. Prepare predict dataset
You can use your own image data (JPG or PNG), or some images from imagenet-2012 validation
dataset <http://image-net.org/download-images> to run the example. We use `/tmp/zoo/infer_images`
in this example.


## Run example
You can run this example on local mode and yarn client mode. 
Note that on local mode you need to ensure environment variable `HADOOP_CONF_DIR` is unset.

- Run with Spark Local mode:
```bash
python imageInference.py \
-m /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model \
-f /tmp/zoo/infer_images \
--b 32
```

- Run with Yarn Client mode:
```bash
# put dataset to hdfs
hdfs dfs -put infer_images_small/ infer_images_small
export HADOOP_CONF_DIR=[path to your hadoop conf directory]

# run example
python imageInference.py \
-m /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model \
-f infer_images_small \
--b 32 \
--cluster_mode yarn
```

In above commands
* `-f` Path to the images.
* `-m` Path to the pre-trained model.
* `--b`, `--batch_size` The number of samples per gradient update. Default is 56.
* `--cluster_mode` The mode of spark cluster, supporting local and yarn. Default is "local".


## Results
You can find the prediction results as:
```
+----------------------------------------------------------------------------------+----------+
|name                                                                              |prediction|
+----------------------------------------------------------------------------------+----------+
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000010.JPEG|283.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000011.JPEG|109.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000012.JPEG|286.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000013.JPEG|370.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000014.JPEG|757.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000015.JPEG|595.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000016.JPEG|147.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000017.JPEG|1.0       |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000018.JPEG|21.0      |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000019.JPEG|478.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000020.JPEG|517.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000021.JPEG|334.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000022.JPEG|179.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000023.JPEG|948.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000024.JPEG|727.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000025.JPEG|23.0      |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000026.JPEG|846.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000027.JPEG|270.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000028.JPEG|166.0     |
|hdfs://172.16.0.165:9000/user/root/infer_images_small/ILSVRC2012_val_00000029.JPEG|64.0      |
+----------------------------------------------------------------------------------+----------+
only showing top 20 rows

```
