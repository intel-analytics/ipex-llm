# Object Detection
There are two simple examples of object detection using Analytics Zoo Object Detection API.
In object-detection.ipynb we use SSD-MobileNet to predict instances of target classes in the given video, which can be regarded as a sequence of images. [The video](https://www.youtube.com/watch?v=akcYAuaP4jw) is a scene of training a dog from ([YouTube-8M dataset](https://research.google.com/youtube8m/)) and the people and the dog are among target classes. Proposed areas are labeled with boxes and class scores.
In messi.ipynb we use a pretrained detect messi model to detect messi in a video. Proposed areas are labeled with boxes and class scores.

## Environment
* Python 3.5/3.6 (Need moviepy)
* Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo)

## Install or download Analytics Zoo
Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run Jupyter after pip install
```bash
export SPARK_DRIVER_MEMORY=8g
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

## Run Jupyter with prebuilt package
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package`.
* Run `pip install moviepy`.
For object-detection.ipynb video can be found from YouTube-8M ([YouTube-8M](https://research.google.com/youtube8m/) and [The video](https://www.youtube.com/watch?v=akcYAuaP4jw)).
Run `$ANALYTICS_ZOO_HOME/apps/object-detection/download_model.sh` to download the pretrained model.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g
```