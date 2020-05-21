<div align="center">
   <p align="center"> <img src="https://github.com/analytics-zoo/analytics-zoo.github.io/blob/master/img/logo.jpg" height=240px; weight=320px;"><br></p>
</div>
      
_A unified Data Analytics and AI platform for **distributed TensorFlow, Keras and PyTorch on Apache Spark/Flink & Ray**_

---

# <font size="6"> What is Analytics Zoo? </font>

Analytics Zoo seamless scales TensorFlow, Keras and PyTorch to distributed big data (using Spark, Flink & Ray).

<div align="center">
   <p align="center"> <img src="docs/docs/Image/blockdiagram.jpg" height=240px; weight=718px;"><br></p>
</div>

* **End-to-end pipeline for applying AI models (TensorFlow, PyTorch, OpenVINO, etc.) to distributed big data**
  * Write [TensorFlow](https://analytics-zoo.github.io/master/#ProgrammingGuide/TFPark/tensorflow/) or [PyTorch](https://analytics-zoo.github.io/master/#ProgrammingGuide/pytorch/) inline with Spark code for distributed training and inference.
  * Native deep learning (TensorFlow/Keras/PyTorch/BigDL) support in [Spark ML](https://analytics-zoo.github.io/master/#ProgrammingGuide/nnframes) Pipelines.
  * Directly run [Ray](https://analytics-zoo.github.io/master/#ProgrammingGuide/rayonspark/) programs on big data cluster through _RayOnSpark_. 
  * Plain Java/Python APIs for (TensorFlow/PyTorch/BigDL/OpenVINO) [Model Inference](https://analytics-zoo.github.io/master/#ProgrammingGuide/inference). 

* **High-level ML workflow for automating machine learning tasks**
  * [Cluster Serving](https://analytics-zoo.github.io/master/#ClusterServingGuide/ProgrammingGuide) for automatically distributed (TensorFlow/PyTorch/Caffe/OpenVINO) model inference . 
  * Scalable [AutoML](https://analytics-zoo.github.io/master/#ProgrammingGuide/AutoML/overview/) for time series prediction.

* **Built-in models** for [Recommendation](https://analytics-zoo.github.io/master/#APIGuide/Models/recommendation/), [Time Series](https://analytics-zoo.github.io/master/#APIGuide/Models/anomaly-detection/), [Computer Vision](https://analytics-zoo.github.io/master/#APIGuide/Models/object-detection/) and [NLP]( https://analytics-zoo.github.io/master/#APIGuide/Models/text-matching/) applications.

---

# <font size="6">Why use Analytics Zoo? </font>

You may want to develop your AI solutions using Analytics Zoo if:
* You want to easily apply AI models (e.g., TensorFlow, Keras, PyTorch, BigDL, OpenVINO, etc.) to distributed big data.
* You want to transparently scale your AI applications from a single laptop to large clusters with "zero" code changes.
* You want to deploy your AI pipelines to existing YARN or K8S clusters *WITHOUT* any modifications to the clusters.
* You want to automate the process of applying machine learning (such as feature engineering, hyperparameter tuning, model selection, distributed inference, etc.). 


---

# <font size="6">How to use Analytics Zoo? </font>
* Check out the [Getting Started page](https://analytics-zoo.github.io/master/#gettingstarted/) for a quick overview of how to use Analytics Zoo.
* Refer to the [Python](https://analytics-zoo.github.io/master/#PythonUserGuide/install/), [Scala](https://analytics-zoo.github.io/master/#ScalaUserGuide/install/) and [Docker](https://analytics-zoo.github.io/master/#DockerUserGuide/) guides to install Analytics Zoo.
* Visit the [Document Website](https://analytics-zoo.github.io/) ([mirror](https://analytics-zoo.gitee.io/) in China) for more information on Analytics Zoo.
* Check the [Powered By](https://analytics-zoo.github.io/master/#powered-by/) & [Presentations](https://analytics-zoo.github.io/master/#presentations/) pages for real-world applications using Analytics Zoo.
* Join the [Google Group](https://groups.google.com/forum/#!forum/bigdl-user-group) (or subscribe to the [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)) for more questions and discussions on Analytics Zoo.
