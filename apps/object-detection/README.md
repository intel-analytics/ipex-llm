# Object Detection
This is a simple example of object detection using Analytics Zoo Object Detection API. We use SSD-MobileNet to predict instances of target classes in the given video, which can be regarded as a sequence of images. [The video](https://www.youtube.com/watch?v=akcYAuaP4jw) is a scene of training a dog from ([YouTube-8M dataset](https://research.google.com/youtube8m/)) and the people and the dog are among target classes. Proposed areas are labeled with boxes and class scores.

## Environment
* Python 2.7/3.5/3.6 (Need moviepy)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)
* Analytics Zoo 0.1.0

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run `pip install moviepy`.
* Prepare the video in YouTube-8M to detect. ([YouTube-8M](https://research.google.com/youtube8m/) and [The video](https://www.youtube.com/watch?v=akcYAuaP4jw)).
* Run `$ANALYTICS_ZOO_HOME/apps/object-detection/download_model.sh` to download the pretrained model.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 2  \
    --driver-memory 8g  \
    --total-executor-cores 2  \
    --executor-cores 2  \
    --executor-memory 8g
```