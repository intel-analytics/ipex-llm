# Orca Bigdl Resnet finetune example

This is an example to demonstrate how to use Analytics-Zoo's Orca Bigdl Estimator API to run distributed [ResNet finetune](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/pytorch/train/resnet_finetune) training and inference task.In this example we use a pre-trained ResNet model, adding an extra layer to the end, to train a dog-vs-cat image classification model.

## Environment Preparation

Download and install latest analytics whl by following instructions ([here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/#install-the-latest-nightly-build-wheels-for-pip)).

```bash
conda create -n zoo python=3.7 # zoo is conda enviroment name, you can set another name you like.
conda activate zoo
pip install analytics-zoo==0.9.0.dev0 # or above
pip install jep==3.9.0 cloudpickle==1.6.0
conda install pytorch torchvision cpuonly -c pytorch # command for linux
conda install pytorch torchvision -c pytorch # command for macOS
```
If java is not installed, use command `java` to check if java is installed, you can use one of following commnads:  
1. system's package management system(like apt): `sudo apt-get install openjdk-8-jdk`.  
2. conda: `conda install openjdk=8.0.152`.
3. Manual installation: [oracle jdk](https://www.oracle.com/java/technologies/javase/javase-jdk8-downloads.html).
Note: conda environment is required to run on Yarn, but not strictly necessary for running on local.

## Data Preparation
For this example we use [Dogs vs. Cats](https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip) train
dataset. Download the data and run the following commands to copy about 222 images of cats
and dogs into `samples` folder.

    ```bash
    wget https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
    unzip cats_and_dogs_filtered.zip
    cd cats_and_dogs_filtered
    mkdir samples
    cp train/cats/cat.7* samples
    cp train/dogs/dog.7* samples
    ```
    `7` is randomly chosen and can be replaced with other digit.

## Image Fine Tuning
You can run this example on local mode and yarn client mode. Note that on local mode you need to ensure environment variable `HADOOP_CONF_DIR` is unset.

- Run with Spark Local mode
```bash
python resnet_finetune.py --imagePath cats_and_dogs_filtered/samples
```

- Run with Yarn Client mode:
```bash
# put dataset to hdfs
hdfs dfs -put cats_and_dogs_filtered/samples dogs_cats 
export HADOOP_CONF_DIR=[path to your hadoop conf directory]

# run example
python resnet_finetune.py --imagePath dogs_cats --cluster_mode yarn
```

Options
* `--imagePath` Path to the images.
* `--cluster_mode` The mode for the Spark cluster. local or yarn. Default to be `local`.

## Results
You can find the logs for training:
```
2021-02-04 06:21:33 INFO  DistriOptimizer$:427 - [Epoch 1 208/205][Iteration 13][Wall Clock 20.492105877s] Trained 16.0 records in 1.507311492 seconds. Throughput is 10.614926 records/second. Loss is 0.27508935. TorchModele924f486's hyper parameters: Current learning rate is 0.001. Current dampening is 1.7976931348623157E308.
```
And after validation, test results will be seen like:
```
+--------------------+-----------+-----+----------+
|               image|       name|label|prediction|
+--------------------+-----------+-----+----------+
|[hdfs://172.16.0....|cat.783.jpg|  1.0|       1.0|
|[hdfs://172.16.0....|cat.795.jpg|  1.0|       1.0|
|[hdfs://172.16.0....|dog.785.jpg|  0.0|       0.0|
+--------------------+-----------+-----+----------+

Validation accuracy = 0.8235294117647058, correct 14,  total 17
```
