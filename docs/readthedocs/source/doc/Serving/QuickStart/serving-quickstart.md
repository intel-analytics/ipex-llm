# BigDL Cluster Serving Quick Start

This section provides a quick start example for you to run BigDL Cluster Serving. To simplify the example, we use docker to run Cluster Serving. If you do not have docker installed, [install docker](https://docs.docker.com/install/) first. The quick start example contains all the necessary components so the first time users can get it up and running within minutes:

* A docker image for BigDL Cluster Serving (with all dependencies installed)
* A sample configuration file
* A sample trained TensorFlow model, and sample data for inference
* A sample Python client program

Use one command to run Cluster Serving container. (We provide quick start model in older version of docker image, for newest version, please refer to following sections and we remove the model to reduce the docker image size).
```
docker run --name cluster-serving -itd --net=host intelanalytics/bigdl-cluster-serving:0.9.1
```
Log into the container using `docker exec -it cluster-serving bash`, and run
```
cd cluster-serving
cluster-serving-init
```
`bigdl.jar` and `config.yaml` is in your directory now.

Also, you can see prepared TensorFlow frozen ResNet50 model in `resources/model` directory with following structure.

```
cluster-serving | 
               -- | model
                 -- frozen_graph.pb
                 -- graph_meta.json
```
Modify `config.yaml` and add following to `model` config
```
model:
    path: resources/model
```

Start Cluster Serving using `cluster-serving-start`. 

Run python program `python3 image_classification_and_object_detection_quick_start.py -i resources/test_image` to push data into queue and get inference result. 

Then you can see the inference output in console. 
```
cat prediction layer shape:  (1000,)
the class index of prediction of cat image result:  292
cat prediction layer shape:  (1000,)
```
Wow! You made it!

Note that the Cluster Serving quick start example will run on your local node only. Check the [Deploy Your Own Cluster Serving](#deploy-your-own-cluster-serving) section for how to configure and run Cluster Serving in a distributed fashion.

For more details, refer to following sections.
