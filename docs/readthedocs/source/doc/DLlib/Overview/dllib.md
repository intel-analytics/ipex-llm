# DLlib User Guide

## 1. Overview

DLlib is a distributed deep learning library for Apache Spark; with DLlib, users can write their deep learning applications as standard Spark programs (using either Scala or Python APIs).

It includes the functionalities of the [original BigDL](https://github.com/intel-analytics/BigDL/tree/branch-0.14) project, and provides following high-level APIs for distributed deep learning on Spark:

* [Keras-like API](keras-api.md) 
* [Spark ML pipeline support](nnframes.md)

## 2. Scala user guide
### 2.1 Install and Run
Please refer [scala guide](../../UserGuide/scala.md) for details.

---

### 2.2 Get started
---

This section show a single example of how to use dllib to build a deep learning application on Spark, using Keras APIs

---
#### **LeNet Model on MNIST using Keras-Style API**

This tutorial is an explanation of what is happening in the [lenet](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/example/keras) example

A bigdl-dllib program starts with initialize as follows.
````scala
      val conf = Engine.createSparkConf()
        .setAppName("Train Lenet on MNIST")
        .set("spark.task.maxFailures", "1")
      val sc = new SparkContext(conf)
      Engine.init
````

After the initialization, we need to:

1. Load train and validation data by _**creating the [```DataSet```](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/dllib/src/main/scala/com/intel/analytics/bigdl/dllib/feature/dataset/DataSet.scala)**_ (e.g., ````SampleToGreyImg````, ````GreyImgNormalizer```` and ````GreyImgToBatch````):

````scala
    val trainSet = (if (sc.isDefined) {
        DataSet.array(load(trainData, trainLabel), sc.get, param.nodeNumber)
      } else {
        DataSet.array(load(trainData, trainLabel))
      }) -> SampleToGreyImg(28, 28) -> GreyImgNormalizer(trainMean, trainStd) -> GreyImgToBatch(
        param.batchSize)

    val validationSet = DataSet.array(load(validationData, validationLabel), sc) ->
        BytesToGreyImg(28, 28) -> GreyImgNormalizer(testMean, testStd) -> GreyImgToBatch(
        param.batchSize)
````

2. We then define Lenet model using Keras-style api
````scala
    val input = Input(inputShape = Shape(28, 28, 1))
    val reshape = Reshape(Array(1, 28, 28)).inputs(input)
    val conv1 = Convolution2D(6, 5, 5, activation = "tanh").setName("conv1_5x5").inputs(reshape)
    val pool1 = MaxPooling2D().inputs(conv1)
    val conv2 = Convolution2D(12, 5, 5, activation = "tanh").setName("conv2_5x5").inputs(pool1)
    val pool2 = MaxPooling2D().inputs(conv2)
    val flatten = Flatten().inputs(pool2)
    val fc1 = Dense(100, activation = "tanh").setName("fc1").inputs(flatten)
    val fc2 = Dense(classNum, activation = "softmax").setName("fc2").inputs(fc1)
    Model(input, fc2)
 ````

3. After that, we configure the learning process. Set the ````optimization method```` and the ````Criterion```` (which, given input and target, computes gradient per given loss function):
````scala
  model.compile(optimizer = optimMethod,
          loss = ClassNLLCriterion[Float](logProbAsInput = false),
          metrics = Array(new Top1Accuracy[Float](), new Top5Accuracy[Float](), new Loss[Float]))
````

Finally we _**train the model**_ by calling ````model.fit````:
````scala
  model.fit(trainSet, nbEpoch = param.maxEpoch, validationData = validationSet)
````

---

## 3. Python user guide

### 3.1 Install

#### 3.1.1 Official Release

Run below command to install _bigdl-dllib_.

```bash
conda create -n my_env python=3.7
conda activate my_env
pip install bigdl-dllib
```

#### 3.1.2 Nightly build

You can install the latest nightly build of bigdl-dllib as follows:
```bash
pip install --pre --upgrade bigdl-dllib
```


### 3.2 Run

#### **3.2.1 Interactive Shell**

You may test if the installation is successful using the interactive Python shell as follows:

* Type `python` in the command line to start a REPL.
* Try to run the example code below to verify the installation:

  ```python
  from bigdl.dllib.utils.nncontext import *

  sc = init_nncontext()  # Initiation of bigdl-dllib on the underlying cluster.
  ```

#### **3.2.2 Jupyter Notebook**

You can start the Jupyter notebook as you normally do using the following command and run bigdl-dllib programs directly in a Jupyter notebook:

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### **3.2.3 Python Script**

You can directly write bigdl-dlllib programs in a Python file (e.g. script.py) and run in the command line as a normal Python program:

```bash
python script.py
```
---
### 3.3 Get started
---

#### **Autograd Examples using bigdl-dllb keras Python API**

This tutorial describes the [Autograd](https://github.com/intel-analytics/BigDL/tree/branch-2.0/python/dllib/examples/autograd).

The example first do the initializton using `init_nncontext()`:
```python
  sc = init_nncontext()
```

It then generate the input data X_, Y_

```python
    data_len = 1000
    X_ = np.random.uniform(0, 1, (1000, 2))
    Y_ = ((2 * X_).sum(1) + 0.4).reshape([data_len, 1])
```

It then define the custom loss

```python
def mean_absolute_error(y_true, y_pred):
    result = mean(abs(y_true - y_pred), axis=1)
    return result
```

After that, the example creates the model as follows and set the criterion as the custom loss:
```python
    a = Input(shape=(2,))
    b = Dense(1)(a)
    c = Lambda(function=add_one_func)(b)
    model = Model(input=a, output=c)

    model.compile(optimizer=SGD(learningrate=1e-2),
                  loss=mean_absolute_error)
```
Finally the example trains the model by calling `model.fit`:

```python
    model.fit(x=X_,
              y=Y_,
              batch_size=32,
              nb_epoch=int(options.nb_epoch),
              distributed=False)
```