## Torch ResNet Prediction Example

TorchNet wraps a Pytorch model as Analytics Zoo module, thus the Pytorch model can be used for
distributed inference. This example illustrates that a PyTorch program, with few lines of change,
can be executed on Apache Spark.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)
to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Model and Data Preparation

We use ResNet 18 from torchvision and run inference on some images, e.g. images from ImageNet.

## Run this example after pip install
```bash
python predict.py --image path_to_image_folder
```

## Run this example with prebuilt package
```bash
export SPARK_HOME=the root directory of Spark
export ANALYTICS_ZOO_HOME=the directory where you extract the downloaded Analytics Zoo zip package
MASTER=... # Spark Master
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    predict.py \
    --image path_to_image_folder 
```

__Options:__
* `--image` The path where the images are stored. 

## Results
The programs outputs the corresponding class for each image.
