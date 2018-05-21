# Variational Autoencoders

This directory contains three notebooks to show you how to use
[Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
in Analytics Zoo. These notebooks are developed on Apache Spark 1.6
and Analytics Zoo 0.1.0.

1. [Using Variational Autoencoder to Generate Digital Numbers](./using_variational_autoencoder_to_generate_digital_numbers.ipynb)
   This notebook used the [MNIST](http://yann.lecun.com/exdb/mnist/)
   dataset as training data, and learned a generative model that we can
   sample hand written digital numbers from.
2. [Using Variational Autoencoder to Generate Human Faces](./using_variational_autoencoder_to_generate_faces.ipynb)
   This notebook use the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
   dataset as training data, and learned a generative model that we can
   sample human faces from.
3. [Using Variational Autoencoder and Deep Feature Loss to Generate Human Faces](./using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.ipynb)
   This notebook is similar to the previous one but use a more sophisticated loss function to generate
   more vivid images at the cost of more training time.

To run this example, you should 
* Download Analytics Zoo and build it.
* Run `export SPARK_HOME=the root directory of Spark`.
* Run `export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project`.
* Run the following bash command to start the jupyter notebook. Change parameter settings as you need, ie `MASTER = local[physcial_core_number]`.
```Bash
MASTER=local[*]
${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
    --master ${MASTER} \
    --driver-cores 4  \
    --driver-memory 22g  \
    --total-executor-cores 4  \
    --executor-cores 4  \
    --executor-memory 22g
```


