# Image Augmentation for 3D Images
This is a simple example of image augmentation for 3D images using Analytics ZOO API. We use various ways to transform images to augment the dataset.

## Environment
* Python 2.7/3.5/3.6 (numpy 1.11.1)
* Apache Spark 1.6.0/2.1.0 (This version needs to be same with the version you use to build Analytics Zoo)

## Run with Jupyter
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 1g  \
    --executor-memory 1g
```
