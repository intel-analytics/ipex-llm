![logo](Image/logo_s.jpg) 

_A unified Data Analytics and AI platform for **distributed TensorFlow, Keras, PyTorch, Apache Spark/Flink and Ray**_

---

# <font size="6"><b>What is Analytics Zoo?</b></font>

Analytics Zoo provides a unified data analytics and AI platform that seamlessly unites TensorFlow, Keras, PyTorch, Spark, Flink and Ray programs into an integrated pipeline, which can transparently scale from a laptop to large clusters to process production big data.



![blockdiagram](Image/blockdiagram.jpg) 


* **Integrated Analytics and AI Pipelines** for easily prototyping and deploying end-to-end AI applications. 
    * Write [TensorFlow](ProgrammingGuide/TFPark/tensorflow.md) or [PyTorch](ProgrammingGuide/pytorch.md) inline with Spark code for distributed training and inference.
    * Native deep learning (TensorFlow/Keras/PyTorch/BigDL) support in [Spark ML](ProgrammingGuide/nnframes.md) Pipelines.
    * Directly run Ray programs on big data cluster through [RayOnSpark](ProgrammingGuide/rayonspark.md). 
    * Plain Java/Python APIs for (TensorFlow/PyTorch/BigDL/OpenVINO) [Model Inference](ProgrammingGuide/inference.md). 

* High-Level **ML Workflow** that automates the process of building large-scale machine learning applications.
    * Automatically distributed [Cluster Serving](ClusterServingGuide/ProgrammingGuide.md) (for TensorFlow/PyTorch/Caffe/BigDL/OpenVINO models) with a simple pub/sub API. 
    * Scalable [AutoML](ProgrammingGuide/AutoML/overview.md) for time series prediction (that automatically generates features, selects models and tunes hyperparameters).

* **Built-in Algorithms and Models** for [Recommendation](APIGuide/Models/recommendation.md), [Time Series](APIGuide/Models/anomaly-detection.md), [Computer Vision](APIGuide/Models/object-detection.md) and [NLP](APIGuide/Models/seq2seq.md) applications.

---

# <font size="6"><b>Why use Analytics Zoo?</b></font>

You may want to develop your AI solutions using Analytics Zoo if:

* You want to easily prototype the entire end-to-end pipeline that applies AI models (e.g., TensorFlow, Keras, PyTorch, BigDL, OpenVINO, etc.) to production big data.
* You want to transparently scale your AI applications from a laptop to large clusters with "zero" code changes.
* You want to deploy your AI pipelines to existing YARN or K8S clusters *WITHOUT* any modifications to the clusters.
* You want to automate the process of applying machine learning (such as feature engineering, hyperparameter tuning, model selection and distributed inference). 

---

# <font size="6"><b>How to use Analytics Zoo?</b></font>

* Check out the [Getting Started page](gettingstarted.md) for a quick overview of how to use Analytics Zoo.
* Refer to the [Python](PythonUserGuide/install.md), [Scala](ScalaUserGuide/install.md) and [Docker](DockerUserGuide/index.md) guides to install Analytics Zoo.
* Visit the [Document Website](https://analytics-zoo.github.io/) ([mirror](https://analytics-zoo.gitee.io/) in China) for more information on Analytics Zoo.
* Check the [Powered By](powered-by.md) & [Presentations](presentations.md) pages for real-world applications using Analytics Zoo.
* Join the [Google Group](https://groups.google.com/forum/#!forum/bigdl-user-group) (or subscribe to the [Mail List](mailto:bigdl-user-group+subscribe@googlegroups.com)) for more questions and discussions on Analytics Zoo.

