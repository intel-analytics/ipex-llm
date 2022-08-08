<div align="center">

<p align="center"> <img src="docs/readthedocs/image/bigdl_logo.jpg" height="140px"><br></p>

</div>

<h3 align="center" style="display: block; font-size: 2.5em; font-weight: bold; margin-block-start: 1em; margin-block-end: 1em;">
For Building Fast, Scalable, and Secured AI 
</h3>

<div align="center">

[![Latest release][release-badge]][release-link] [![Latest release date][release-date-badge]][release-link] [![PyPI][pypi-badge]][pypi-link] 

</div>

<p align="center">
	<strong>
		<a href="https://bigdl.readthedocs.io/">Docs</a>
		•
		<a href="https://arxiv.org/ftp/arxiv/papers/2204/2204.01715.pdf/">Paper</a>
		•
		<a href="https://huggingface.co/spaces/BigDL/bigdl_nano_demo">Demo</a>
    		•
		<a href="https://bigdl.readthedocs.io/en/latest/doc/UserGuide/docker.html">Docker</a>
	</strong>
</p>



---


**BigDL is a suite of libraries which can help data scientists and engineers to easily build end-to-end, fast, and scalable AI applications in various aspects.** Each library in BigDL can be installed and used alone or in combinition:

* [_Nano_]() - transparent acceleration of PyTorch/Tensorflow.
* [_Orca_]() - seamless scaling local AI to cluster.
* [_DLLib_]() - deep learning for Apache Spark
* [_BigDL-Chronos_]() - fast and scalable time series analysis.
* [_BigDL-Friesian_]() - large-scale recommendation systems.
* [_BigDL-PPML_]() - secure and privacy preserved AI
 


## What can you do with BigDL


- ✅ make your single-node Tensorflow/PyTroch programs faster - Refer to [_Nano_](https://bigdl.readthedocs.io/en/latest/doc/Nano/Overview/nano.html) <br/>
- ✅ turn your single-node AI applications to distributed and run on a large cluster - Refer to [_Orca: Distributed Training and Inference_](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-training-inference.html#) <br/>
- ✅ use automatic hyperparameter tuning in parallel or in a distributed cluster - Refer to [_Orca: Distributed Hyperparam Tuning_](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-tuning.html) <br/>
- ✅ develop deep learning applications in Scala or Python from scratch to run on Spark cluster - Refer to [_DLLib_](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/dllib.html)  <br/>
- ✅ run Ray applications on Spark cluster - Refer to [_Orca: RayOnSpark_](https://bigdl.readthedocs.io/en/latest/doc/Ray/Overview/ray.html)  <br/>
- ✅ build fast and scalable domain-specific applications optimized on Xeon platforms (e.g. Time series, Recommendation Systems) - Refer to [_Chronos_](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html) and [_Friesian_]()  <br/>
- ✅ run machine learning and deep learning in securied and privacy preserved environment [_PPML_](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) <br/>

<table align="center">
	<tr><td><details>
		<summary> Refer to <em>Nano</em> </summary>
			✅ make your single-node Tensorflow/PyTroch programs faster - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/Nano/Overview/nano.html">Nano</a> <br/>
	</details></td>
	<td><details>
		<summary> Refer to <em>Orca</em> </summary>
			✅ turn your single-node AI applications to distributed and run on a large cluster - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-training-inference.html">Orca: Distributed Training and Inference</a><br/>
			✅ use automatic hyperparameter tuning in parallel or in a distributed cluster - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-tuning.html">Orca: Distributed Hyperparam Tuning</a><br/>
		✅ run Ray applications on Spark cluster - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/Ray/Overview/ray.html">Orca: RayOnSpark</a>  <br/>
	</details></td>
	<td><details>
		<summary> Refer to <em>DLLib</em> </summary>
			✅ develop deep learning applications in Scala or Python from scratch to run on Spark cluster - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/dllib.html">DLLib</a> <br/>
	</details></td>
	<td><details>
		<summary> Refer to <em>Chronos and Friesian</em> </summary>
			✅ build fast and scalable domain-specific applications optimized on Xeon platforms (e.g. Time series, Recommendation Systems) - Refer to <a href="https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html">Chronos</a> and <a href=#>Friesian</a>  <br/>
	</details></td>
	<td><details>
		<summary> Refer to <em>PPML</em> </summary>
			✅ run machine learning and deep learning in securied and privacy preserved environment <a href="https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html">PPML</a> <br/>
	</details></td></tr>
</table>

BigDL can make all above tasks easy. For more guidence, tutorials and demos, refer to our [Docs site](https://bigdl.readthedocs.io/).

---

## Installation
- You can install a stable release of BigDL using pip (including all libraries)
	```bash
	pip install bigdl
	```
   For more information, refer to [Python user guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html). Scala users, please refer to [Scala user guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/scala.html)

- Or you may want to install individual libraries. For example, install chronos only 
	```
	pip install bigdl-chronos
	```
  The installtion steps may differ for each library. Please refer to the installation guide for more specifics. 

---

## Quick Usage Demo 

### Nano

Using *BigDL-Nano*, you can acclerate your local PyTorch or Tensorflow program, with only miniumum change of code. 
<details><summary>Show Nano example</summary>
<br/>
First, import bigdl nano trainer.

```python
from bigdl.nano.pytorch.trainer import Trainer
```

Then, load model and define data loader as in standard pytorch code.
```python 
# load model
device = 'cpu'
dtype = torch.float32
model = torch.load("models/generator.pt")
model.eval()
model.to(device, dtype)

# define loader
loader = torch.utils.data.DataLoader(...)
```

Before inference, use trace to get an accelerated model. 
model = Trainer.trace(model, accelerator='openvino', input_sample=next(iter(loader)))

Finally, do inference using the model the same way as in standard pytorch code. 
```python
with torch.no_grad():
    for inputs in tqdm(loader):
        inputs = inputs.to(device, dtype)
        outputs = model(inputs)
```

</details>

### Orca

Using *BigDL-Orca*, you can scale out local _**TensorFlow**_ or _**PyTorch**_ applications end-to-end (i.e. training, inference, data processing) seamlessly across large clusters.

<details><summary>Show Orca example</summary>
<br/>
First, initialize [Orca Context](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html):

```python
from bigdl.orca import init_orca_context, OrcaContext

# cluster_mode can be "local", "k8s" or "yarn"
sc = init_orca_context(cluster_mode="yarn", cores=4, memory="10g", num_nodes=2) 
```

Next, perform [data-parallel processing in Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/data-parallel-processing.html) (supporting standard Spark Dataframes, TensorFlow Dataset, PyTorch DataLoader, Pandas, Pillow, etc.):

```python
from pyspark.sql.functions import array

spark = OrcaContext.get_spark_session()
df = spark.read.parquet(file_path)
df = df.withColumn('user', array('user')) \  
       .withColumn('item', array('item'))
```

Finally, use [sklearn-style Estimator APIs in Orca](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/distributed-training-inference.html) to perform distributed _TensorFlow_, _PyTorch_ or _Keras_ training and inference:

```python
from tensorflow import keras
from bigdl.orca.learn.tf.estimator import Estimator

user = keras.layers.Input(shape=[1])  
item = keras.layers.Input(shape=[1])  
feat = keras.layers.concatenate([user, item], axis=1)  
predictions = keras.layers.Dense(2, activation='softmax')(feat)  
model = keras.models.Model(inputs=[user, item], outputs=predictions)  
model.compile(optimizer='rmsprop',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy'])

est = Estimator.from_keras(keras_model=model)  
est.fit(data=df,  
        batch_size=64,  
        epochs=4,  
        feature_cols=['user', 'item'],  
        label_cols=['label'])
```

See [TensorFlow](https://bigdl.readthedocs.io/en/latest/doc/Orca/QuickStart/orca-tf-quickstart.html) and [PyTorch](https://bigdl.readthedocs.io/en/latest/doc/Orca/QuickStart/orca-pytorch-quickstart.html) quickstart, as well as the [document website](https://bigdl.readthedocs.io/), for more details.

</details>


### DLlib

Using *BigDL-DLlib*, you can write distributed deep learning applications as standard Spark programs (using either Scala or Python APIs).

<details><summary>Show DLLib example</summary>
<br/>
First, call `initNNContext` at the beginning of the code: 

```scala
import com.intel.analytics.bigdl.dllib.NNContext
val sc = NNContext.initNNContext()
```

Then, define the BigDL model using Keras-style API:

```scala
val input = Input[Float](inputShape = Shape(10))  
val dense = Dense[Float](12).inputs(input)  
val output = Activation[Float]("softmax").inputs(dense)  
val model = Model(input, output)
```

After that, use `NNEstimator` to train/predict/evaluate the model using Spark Dataframes and ML pipelines:

```scala
val trainingDF = spark.read.parquet("train_data")
val validationDF = spark.read.parquet("val_data")
val scaler = new MinMaxScaler().setInputCol("in").setOutputCol("value")
val estimator = NNEstimator(model, CrossEntropyCriterion())  
        .setBatchSize(size).setOptimMethod(new Adam()).setMaxEpoch(epoch)
val pipeline = new Pipeline().setStages(Array(scaler, estimator))

val pipelineModel = pipeline.fit(trainingDF)  
val predictions = pipelineModel.transform(validationDF)
```
See the [NNframes](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/nnframes.html) and [Keras API](https://bigdl.readthedocs.io/en/latest/doc/DLlib/Overview/keras-api.html) user guides for more details.

</details>

### Chronos 

With *BigDL-Chronos*, you can easily build fast, accurate, scalable time series forecasting and anomaly detection applications (e.g. using built-in advanced deep learning models, built-in inference acclerations, AutoML support, data processing and feature generaition tools, etc.).

<details><summary>Show Chronos example</summary>
<br/>
To train a time series model with AutoML, first initialize [Orca Context](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html):

```python
from bigdl.orca import init_orca_context

#cluster_mode can be "local", "k8s" or "yarn"
init_orca_context(cluster_mode="yarn", cores=4, memory="10g", num_nodes=2, init_ray_on_spark=True)
```

Then, create _TSDataset_ for your data.
```python
from bigdl.chronos.data import TSDataset

tsdata_train, tsdata_valid, tsdata_test\
        = TSDataset.from_pandas(df, 
                                dt_col="dt_col", 
                                target_col="target_col", 
                                with_split=True, 
                                val_ratio=0.1, 
                                test_ratio=0.1)
```

Next, create an _AutoTSEstimator_.

```python
from bigdl.chronos.autots import AutoTSEstimator

autotsest = AutoTSEstimator(model='lstm')
```

Finally, call ```fit``` on _AutoTSEstimator_, which applies AutoML to find the best model and hyper-parameters; it returns a _TSPipeline_ which can be used for prediction or evaluation.

```python
#train a pipeline with AutoML support
ts_pipeline = autotsest.fit(data=tsdata_train,
                            validation_data=tsdata_valid)

#predict
ts_pipeline.predict(tsdata_test)
```

See the Chronos [user guide](https://bigdl.readthedocs.io/en/latest/doc/Chronos/Overview/chronos.html) and [example](https://bigdl.readthedocs.io/en/latest/doc/Chronos/QuickStart/chronos-autotsest-quickstart.html) for more details.

</details>

### Friesian
With *BigDL-Friesian*, you can easily build large-scale recommender application optimized on Intel Xeon.

See the [Friesian user guide]() for more details.

### PPML

With *BigDL-PPML*, you can run unmodified Big Data analysis and ML/DL programs (such as Apache Spark, Apache Flink, Tensorflow, PyTorch, etc.) in a secure fashion on (private or public) cloud.

See the [PPML user guide](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) for more details. 

---

## Getting Support

- [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)
- [User Group](https://groups.google.com/forum/#!forum/bigdl-user-group)
- [Github Issues](https://github.com/intel-analytics/BigDL/issues)

---

## Note
As of **BigDL 2.0** release, we combine the [original BigDL](https://github.com/intel-analytics/BigDL/tree/branch-0.14) and [Analytics Zoo](https://github.com/intel-analytics/analytics-zoo) projects into a single project.  

## Citation

If you've found BigDL useful for your project, you may cite the [paper](https://arxiv.org/ftp/arxiv/papers/2204/2204.01715.pdf) as follows:

```
@inproceedings{dai2022bigdl,
  title={BigDL 2.0: Seamless Scaling of AI Pipelines from Laptops to Distributed Cluster},
  author={Dai, Jason Jinquan and Ding, Ding and Shi, Dongjie and Huang, Shengsheng and Wang, Jiao and Qiu, Xin and Huang, Kai and Song, Guoqiong and Wang, Yang and Gong, Qiyuan and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21439--21446},
  year={2022}
}
```

[release-badge]: https://img.shields.io/github/v/release/intel-analytics/BigDL?label=%20%F0%9F%93%A3%20Latest%20release&style=flat&logoColor=b0c0c0&labelColor=363D44
[release-link]: https://github.com/intel-analytics/BigDL/releases
[release-date-badge]: https://img.shields.io/github/release-date/intel-analytics/BigDL?label=Latest%20release%20date
[pypi-badge]: https://img.shields.io/pypi/v/bigdl.svg
[pypi-link]: https://pypi.org/project/bigdl
