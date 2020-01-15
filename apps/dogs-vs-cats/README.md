# Transfer Learning
In this notebook, we will use a pre-trained Inception_V1 model. But we will operate on the pre-trained model to freeze first few layers, replace the classifier on the top, then fine tune the whole model. And we use the fine-tuned model to solve the dogs-vs-cats classification problem,

## Environment
* Python 3.5/3.6
* JDK 8
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)
* Jupyter Notebook 4.1

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run Jupyter after pip install
```bash
export SPARK_DRIVER_MEMORY=10g
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package`.
* Prepare the training dataset from https://www.kaggle.com/c/dogs-vs-cats and extract it.
  
  The following commands copy about 1100 images of cats and dogs into demo/cats and demo/dogs separately.
```bash          
mkdir -p demo/dogs
mkdir -p demo/cats
cp train/cat.7* demo/cats
cp train/dog.7* demo/dogs
```

* Download the pre-trained [Inception-V1 model](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model). Alternatively, user may also download pre-trained caffe/Tensorflow/keras model.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[4]
bash ${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
        --master ${MASTER} \
        --driver-cores 1  \
        --driver-memory 10g  \
        --total-executor-cores 1  \
        --executor-cores 1  \
        --executor-memory 10g
```
