## Summary

Python demo of transfer Learning based on Spark DataFrame (Dataset). 

Analytics Zoo provides the DataFrame-based API for image reading, preprocessing, model training
and inference. The related classes followed the typical estimator/transformer pattern of Spark ML
and can be used in a standard Spark ML pipeline.

In this example, we will show you how to use a pre-trained inception-v1 model trained on
imagenet dataset to solve the dogs-vs-cats classification problem by transfer learning with
Analytics Zoo. For transfer learning, we will treat the inception-v1 model as a feature extractor
and only train a linear model on these features.

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/)
to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Image Transfer Learning
1. For this example we use kaggle [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) train
dataset. Download the data and run the following commands to copy about 1100 images of cats
and dogs into `samples` folder.

    ```bash
    unzip train.zip -d /tmp/zoo/dogs_cats
    cd /tmp/zoo/dogs_cats
    mkdir samples
    cp train/cat.7* samples
    cp train/dog.7* samples
    ```
    `7` is randomly chosen and can be replaced with other digit.

2. Get the pre-trained Inception-V1 model
Download the pre-trained Inception-V1 model from [Analytics Zoo](https://s3-ap-southeast-1.amazonaws.com/bigdl-models/imageclassification/imagenet/bigdl_inception-v1_imagenet_0.4.0.model),
and put it in `/tmp/zoo` or other path.

3. Run the image transfer learning:
ImageTransferLearningExample.py takes 2 parameters: Path to the pre-trained models and 
Path to the images.

- Run after pip install
You can easily use the following commands to run this example:
    ```bash
    export SPARK_DRIVER_MEMORY=5g
    python ImageTransferLearningExample.py /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

- Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:
    ```bash
    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

    ${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master local[1] \
    --driver-memory 5g \
    ImageTransferLearningExample.py \
    /tmp/zoo/bigdl_inception-v1_imagenet_0.4.0.model /tmp/zoo/dogs_cats/samples
    ```
    See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

4. see the result
After training, you should see something like this in the console:
    ```
    +--------------------+------------+-----+--------------------+----------+
    |               image|        name|label|           embedding|prediction|
    +--------------------+------------+-----+--------------------+----------+
    |[file:/tmp/zoo/do...|cat.7294.jpg|  1.0|[6.7788767E-7, 4....|       1.0|
    |[file:/tmp/zoo/do...|cat.7353.jpg|  1.0|[4.956814E-6, 2.9...|       1.0|
    |[file:/tmp/zoo/do...|cat.7363.jpg|  1.0|[3.5506052E-6, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7464.jpg|  1.0|[3.1471086E-6, 1....|       1.0|
    |[file:/tmp/zoo/do...|cat.7741.jpg|  1.0|[4.4906E-5, 5.736...|       1.0|
    |[file:/tmp/zoo/do...|cat.7798.jpg|  1.0|[5.948801E-6, 8.0...|       1.0|
    |[file:/tmp/zoo/do...|cat.7806.jpg|  1.0|[1.0959853E-6, 5....|       1.0|
    |[file:/tmp/zoo/do...|cat.7909.jpg|  1.0|[4.113644E-5, 1.8...|       1.0|
    |[file:/tmp/zoo/do...|dog.7051.jpg|  2.0|[2.739595E-7, 2.4...|       2.0|
    |[file:/tmp/zoo/do...|dog.7070.jpg|  2.0|[4.9666202E-8, 2....|       2.0|
    |[file:/tmp/zoo/do...|dog.7200.jpg|  2.0|[1.8055023E-4, 3....|       2.0|
    |[file:/tmp/zoo/do...|dog.7320.jpg|  2.0|[0.0010374242, 2....|       2.0|
    |[file:/tmp/zoo/do...|dog.7329.jpg|  2.0|[9.2436676E-5, 1....|       2.0|
    |[file:/tmp/zoo/do...|dog.7494.jpg|  2.0|[2.0494679E-6, 6....|       2.0|
    |[file:/tmp/zoo/do...|dog.7825.jpg|  2.0|[1.9400559E-6, 6....|       2.0|
    |[file:/tmp/zoo/do...|dog.7833.jpg|  2.0|[1.7606219E-5, 9....|       2.0|
    |[file:/tmp/zoo/do...| dog.784.jpg|  2.0|[4.171166E-4, 5.3...|       2.0|
    |[file:/tmp/zoo/do...|dog.7991.jpg|  2.0|[4.5410037E-8, 3....|       2.0|
    +--------------------+------------+-----+--------------------+----------+
    
    Test Error = 0.0333333
    ```
    With master = local[1]. The transfer learning can finish in 8 minutes. As we can see,
    the model from transfer learning can achieve high accuracy on the validation set.
