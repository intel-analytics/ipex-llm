---
## **Before using BigDL**

Before using BigDL, you need to install Apache Spark and obtain BigDL libraries. Then in your program, you need to ensure the SparkContext is created successfully and initialize BigDL engine before calling BigDL APIs. Navigate to [Scala User Guide/Install](ScalaUserGuide/install-pre-built.md) or [Python User Guide/Install](PythonUserGuide/install-from-pip.md) for details about how to install BigDL, and [Scala User Guide/Run](ScalaUserGuide/run.md) or [Python User Guide/Run](PythonUserGuide/run-from-pip.md) for how to run programs. 

---

## **Prepare your Data**

Your data need to be transformed into RDD of [Sample](APIGuide/Data.md#sample) in order to be fed into BigDL for training, evaluation and prediction (also refer to [Optimization](ProgrammingGuide/optimization.md) and [Optimizer API guide](APIGuide/Optimizers/Optimizer.md)). 

[Tensor](APIGuide/Data.md#tensor), [Table](APIGuide/Data.md#table) are essential data structures that composes the basic dataflow inside the neural network( e.g. input/output, gradients, weights, etc.). You will need to understand them to get a better idea of layer behaviors. 


---

## **Use BigDL for Prediction only**

If you have an existing model and want to use BigDL only for prediction, you need first load the model, and then do prediction or evaluation. 

BigDL supports loading models trained and saved in BigDL, or a trained Tensorflow, Caffe or Keras model. 

* To load a BigDL model, you can use `Module.load` interface (Scala) or `Model.load` (in Python). Refer to [Model Save](APIGuide/Module/#model-save) for details.  
* To load a Tensorflow model, refer to [Tensorflow Support](ProgrammingGuide/tensorflow-support.md) for details.
* To load a Caffe model, refer to [Caffe Support](ProgrammingGuide/caffe-support.md) for details.
* To load a Keras model, refer to [Keras Support](ProgrammingGuide/keras-support.md) for details.

Refer to [Model Predict](APIGuide/Module/#model-prediction) for details about how to use a model for prediction.

If you are using the trained model as a component inside a Spark ML pipeline, refer to
[Using BigDL in Spark ML Pipeline](ProgrammingGuide/MLPipeline.md) page for usage. 

---

## **Train a Model from Scratch**

The procedure of training a model from scratch usually involves following steps:

1. define your model (by connecting layers/activations into a network)
2. decide your loss function (which function to optimize)
3. optimization (choose a proper algorithm and hyper parameters, and train)
4. evaluation (evaluate your model) 

Before training models, please make sure BigDL is installed, BigDL engine initialized properly, and your data is in proper format. Refer to [Before using BigDL](#before-using-bigdl) and [Prepare Your Data](#prepare-your-data) for details.  

The most recommended way to create your first model is to modify from an existing one. BigDL provides plenty of models for you to refer to. See [Scala Models/Examples](ScalaUserGuide/resources.md) and [Python Models/Examples and Tutorials](PythonUserGuide/python-resources.md). 

To define a model, you can either use the Sequential API or Functional API. The Functional API is more flexible than Sequential API. Refer to [Sequential API](ProgrammingGuide/Model/Sequential.md) and [Functional API](ProgrammingGuide/Model/Functional.md) for how to define models in different shapes. Navigate to *API Guide/Layers* on the side bar to find the documenations of available layers and activation.

After creating the model, you will have to decide which loss function to use in training. Find the details of losses defined in BigDL in [Losses](APIGuide/Losses.md).  

Now you create an `Optimizer` and set the loss function, input dataset along with other hyper parameters into the Optimizer. Then call `Optimizer.optimize` to train. Refer to [Optimization](ProgrammingGuide/optimization.md) and [Optimizer API guide](APIGuide/Optimizers/Optimizer.md) for details. 

Model Evaluation can be performed periodically during a training. Refer to [Validate your Model in Training](ProgrammingGuide/optimization.md#validate-your-model-in-training) for details.  For a list of defined metrics, refer to [Metrics](APIGuide/Metrics.md).

When `Optimizer.optimize` finishes, it will return a trained model. You can then use the trained model for prediction or evaluation. Refer to [Model Prediction](APIGuide/Module.md#model-prediction) and [Model Evaluation](APIGuide/Module.md#model-evaluation) for detailed usage.    

If you prefer to train a model inside a Spark ML pipeline, please refer to  [Using BigDL in Spark ML Pipeline](ProgrammingGuide/MLPipeline.md) page for usage.

---

## **Save a Model**

When training is finished, you may need to save the final model for later use. 

BigDL allows you to save your BigDL model on local filesystem, HDFS, or Amazon s3 (refer to [Model Save](APIGuide/Module.md#model-save)). 

You may also save the model to Tensorflow or Caffe format (refer to [Caffe Support](ProgrammingGuide/caffe-support.md), and [Tensorflow Support](ProgrammingGuide/tensorflow-support.md) respectively).  

---

## **Stop and Resume a Training**

Training a deep learning model sometimes takes a very long time. It may be stopped or interrupted and we need the training to resume from where we have left. 

To enable this, you have to configure `Optimizer` to periodically take snapshots of the model (trained weights, biases, etc.) and optim-method (configurations and states of the optimization) and dump them into files. Refer to [Checkpointing](ProgrammingGuide/optimization/#checkpointing) for details. 

To resume a training after it stops, refer to [Resume Training](ProgrammingGuide/optimization.md#resume-training).
 

--- 

## **Use Pre-trained Models/Layers**

Pre-train is a useful strategy when training deep learning models. You may use the pre-trained features (e.g. embeddings) in your model, or do a fine-tuning for a different dataset or target.
 
To use a learnt model as a whole, you can use `Module.load` to load the entire model, Then create an `Optimizer` with the loaded model set into it. Refer to [Optmizer API](APIGuide/Optimizers/Optimizer.md) and [Module API](APIGuide/Module.md) for details. 

Instead of using an entire model, you can also use pre-trained weights/biases in certain layers. After a layer is created, use `setWeightsBias` (in Scala) or `set_weights` (in Python) on the layer to initialize the weights with pre-trained weights. Then continue to train your model as usual. 


---

## **Monitor your training**


BigDL provides a convenient way to monitor/visualize your training progress. It writes the statistics collected during training/validation and they can be visualized in real-time using tensorboard. These statistics can also be retrieved into readable data structures later and visualized in other tools (e.g. Jupyter notebook). For details, refer to [Visualization](ProgrammingGuide/visualization.md). 

---

## **Tuning**

There're several strategies that may be useful when tuning an optimization. 

 * Change the learning Rate Schedule in SGD. Refer to [SGD docs](APIGuide/Optimizers/Optim-Methods.md#sgd) for details. 
 * If overfit is seen, try use Regularization. Refer to [Regularizers](APIGuide/Regularizers.md). 
 * Try change the initialization methods. Refer to [Initailizers](APIGuide/Initializers.md).
 * Try Adam or Adagrad at the first place. If they can't achieve a good score, use SGD and find a proper learning rate schedule - it usually takes time, though. RMSProp is recommended for RNN models. Refer to [Optimization Algorithms](APIGuide/Optimizers/Optim-Methods.md) for a list of supported optimization methods. 
