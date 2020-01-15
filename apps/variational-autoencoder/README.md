# Variational Autoencoders

This directory contains three notebooks to show you how to use
[Variational Autoencoders](https://arxiv.org/pdf/1606.05908.pdf)
in Analytics Zoo. These notebooks are developed on Apache Spark 2.x (This version needs to be same with the version you use to build Analytics Zoo).

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

## Install or download Analytics Zoo  
* Follow the instructions [here](https://analytics-zoo.github.io/master/#PythonUserGuide/install/) to install analytics-zoo via __pip__ or __download the prebuilt package__.

## Run after pip install
You can easily use the following commands to run this example:

    export SPARK_DRIVER_MEMORY=22g
    jupyter notebook --notebook-dir=./ --ip=* --no-browser

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-after-pip-install) for more running guidance after pip install.

## Run with prebuilt package
Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode:

    export SPARK_HOME=the root directory of Spark
    export ANALYTICS_ZOO_HOME=the folder where you extract the downloaded Analytics Zoo zip package

Run the following bash command to start the jupyter notebook. Change parameter settings as you need, e.g. `MASTER = local[physcial_core_number]`.

	MASTER=local[*]
	${ANALYTICS_ZOO_HOME}/bin/jupyter-with-zoo.sh \
		--master ${MASTER} \
		--driver-cores 4  \
		--driver-memory 22g  \
		--total-executor-cores 4  \
		--executor-cores 4  \
		--executor-memory 22g

See [here](https://analytics-zoo.github.io/master/#PythonUserGuide/run/#run-without-pip-install) for more running guidance without pip install.

