# Variational Autoencoders

This directory contains three notebooks to show you how to use
[Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
in BigDL. These notebooks are developed on Apache Spark 2.1
and BigDL 0.5.0 snapshot.

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

To run this example, you should first build the BigDL jar or download the pre-built version from [here](https://bigdl-project.github.io/master/#release-download/)
and set the `SPARK_HOME` and `BIGDL_HOME` variable in the `start_notebook.sh` script. Then use this script to
start the notebook.


