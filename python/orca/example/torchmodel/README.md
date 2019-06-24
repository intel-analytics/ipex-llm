## Torch ResNet Prediction Example

TorchNet wraps a TorchScript model as a single layer, thus the Pytorch model can be used for
distributed inference. This example illustrates that a PyTorch program, with One line of change,
can be executed on Apache Spark.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model and Data Preparation

1. Prepare the image dataset for inference. Put the images to do prediction in the same folder.


## Run this example after pip install
```bash
python predict.py --image path_to_image_folder
```

__Options:__
* `--image` The path where the images are stored. 

## Run this example with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the directory where you extract the downloaded Analytics Zoo zip package
MASTER=... # Spark Master
${ANALYTICS_ZOO_HOME}/bin/spark-submit-with-zoo.sh \
    --master ${MASTER} \
    predict.py \
    --image path_to_image_folder 
```

__Options:__
* `--image` The path where the images are stored. 

## Results
The programs outputs the corresponding class for each image.
