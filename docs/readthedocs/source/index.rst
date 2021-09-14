Analytics Zoo Documentation
===========================

------

`Analytics Zoo <https://github.com/intel-analytics/analytics-zoo>`_ is an open source Big Data AI platform, and includes the following features for scaling end-to-end AI to distributed Big Data:

* `Orca <doc/Orca/Overview/orca.html>`_: seamlessly scale out TensorFlow and PyTorch for Big Data (using Spark & Ray)
* `RayOnSpark <doc/Ray/Overview/ray.html>`_: run Ray programs directly on Big Data clusters
* **BigDL Extensions**: high-level `Spark ML pipeline <doc/UseCase/nnframes.html>`_ and `Keras-like <doc/UseCase/keras-api.html>`_ APIs for BigDL 
* `Chronos <doc/Chronos/Overview/chronos.html>`_: scalable time series analysis using AutoML
* `PPML <doc/PPML/Overview/ppml.html>`_: privacy preserving big data analysis and machine learning (*experimental*)

 
-------


.. meta::
   :google-site-verification: hG9ocvSRSRTY5z8g6RLn97_tdJvYRx_tVGhNdtZZavM

.. toctree::
   :maxdepth: 1
   :caption: Quick Start

   doc/Orca/QuickStart/orca-tf-quickstart.md
   doc/Orca/QuickStart/orca-keras-quickstart.md
   doc/Orca/QuickStart/orca-tf2keras-quickstart.md
   doc/Orca/QuickStart/orca-pytorch-quickstart.md
   doc/Ray/QuickStart/ray-quickstart.md

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   doc/UserGuide/python.md
   doc/UserGuide/scala.md
   doc/UserGuide/colab.md
   doc/UserGuide/docker.md
   doc/UserGuide/hadoop.md
   doc/UserGuide/k8s.md
   doc/UserGuide/databricks.md
   doc/Ray/Overview/ray.md
   doc/Chronos/Overview/chronos.md
   doc/PPML/Overview/ppml.md
   doc/UserGuide/develop.md
   
.. toctree::
   :maxdepth: 1
   :caption: Common Use Case
   
   doc/Orca/QuickStart/orca-pytorch-distributed-quickstart.md
   doc/UseCase/spark-dataframe.md
   doc/UseCase/xshards-pandas.md
   doc/Chronos/QuickStart/chronos-autotsest-quickstart.md
   doc/Chronos/QuickStart/chronos-tsdataset-forecaster-quickstart.md
   doc/Chronos/QuickStart/chronos-anomaly-detector.md
   doc/UseCase/keras-api.md
   doc/UseCase/nnframes.md
   doc/Orca/QuickStart/orca-autoestimator-pytorch-quickstart.md
   doc/Orca/QuickStart/orca-autoxgboost-quickstart.md
   
.. toctree::
   :maxdepth: 1
   :caption: Orca Overview

   doc/Orca/Overview/orca.md
   doc/Orca/Overview/orca-context.md
   doc/Orca/Overview/data-parallel-processing.md
   doc/Orca/Overview/distributed-training-inference.md
   doc/Orca/Overview/distributed-tuning.md

.. toctree::
   :maxdepth: 1
   :caption: Python API
   
   doc/PythonAPI/Orca/orca.rst
   doc/PythonAPI/AutoML/automl.rst
   doc/PythonAPI/Friesian/feature.rst
   doc/PythonAPI/Chronos/index.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Real-World Application
   
   doc/Application/presentations.md
   doc/Application/powered-by.md  
