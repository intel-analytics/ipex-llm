# Orca OpenVINO Estimator Inference example
We demonstrate how to use orca OpenVINO estimator to run inference.

## Prepare the environment

We recommend you to use Anaconda to prepare the environment, especially if you want to run on a yarn cluster:

[OpenVINO System requirements](https://software.intel.com/en-us/openvino-toolkit/documentation/system-requirements):

    Ubuntu 18.04 LTS (64 bit)
    Ubuntu 20.04 LTS (64 bit) preview support
    CentOS 7 (64 bit)
    Red Hat* Enterprise Linux* 8 (64 bit)
    macOS 10.15 (64 bit)

```
conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
conda activate zoo
pip install filelock packaging psutil opencv-python
pip install --pre --upgrade bigdl-dllib bigdl-orca # download and install latest nightly-build bigdl-dllib and bigdl-orca.
```

Then install OpenVINO from Anaconda. See [here](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_conda.html)

## Model and Data Preparation
1. Prepare a pre-trained TensorFlow object detection model.

In this example, we use `yolo-v3-tf.xml` and `yolo-v3-tf.bin` from [yolo-v3-tf](https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/yolo-v3-tf). Please put these two files in the same directory.


2. Prepare the image dataset in jpg format for inference. Put the images to do prediction in the same folder.

## Run on local after pip install
```
python predict.py --model_path /path/to/the/OpenVINO/model/yolo-v3-tf.xml --image_folder /path/to/the/image/folder
```

## Run on yarn cluster for yarn-client mode after pip install
```
export HADOOP_CONF_DIR=the directory of the hadoop and yarn configurations
python predict.py --cluster_mode yarn --model_path /path/to/the/OpenVINO/model/yolo-v3-tf.xml --image_folder /path/to/the/image/folder
```

## Other parameters
* --cluster_mode: The cluster mode, such as local, yarn.
* --model_path: Path to the OpenVINO model file.
* --image_folder: The path to the folder where the images are stored.
* --core_num: The number of cpu cores you want to use on each node. You can change it depending on your own cluster setting.
* --executor_num: The number of executors when cluster_mode=yarn.
* --data_num: The number of dummy data.
* --batch_size: The batch size of inference.
* --memory: The executor memory size.
