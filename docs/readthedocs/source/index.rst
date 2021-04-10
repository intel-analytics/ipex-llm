Analytics Zoo Documentation
===========================

------

Analytics Zoo is an open source Big Data AI platform, and includes the following features for scaling end-to-end AI to distributed Big Data:

* `Orca <./doc/Orca/Overview/orca.md>`_: seamlessly scale out TensorFlow and PyTorch programs for Big Data (using Spark & Ray)
* `RayOnSpark <./doc/Ray/Overview/ray.md>`_: run Ray programs directly on Big Data clusters
* **BigDL Extensions**: high-level `Spark ML pipeline <./doc/UseCase/nnframes.md>`_ and `Keras-like <./doc/UseCase/keras-api.md>`_ APIs for BigDL 
* `Zouwu <./doc/Zouwu/Overview/zouwu.md>`_: scalable time series analysis using AutoML
 
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
   doc/Zouwu/Overview/zouwu.md
   doc/UserGuide/develop.md
   
.. toctree::
   :maxdepth: 1
   :caption: Common Use Case
   
   doc/Orca/QuickStart/orca-pytorch-distributed-quickstart.md
   doc/UseCase/spark-dataframe.md
   doc/UseCase/xshards-pandas.md
   doc/Zouwu/QuickStart/zouwu-autots-quickstart.md
   doc/UseCase/keras-api.md
   doc/UseCase/nnframes.md
   
.. toctree::
   :maxdepth: 1
   :caption: Orca Overview

   doc/Orca/Overview/orca-context.md
   doc/Orca/Overview/data-parallel-processing.md
   doc/Orca/Overview/distributed-training-inference.md

.. toctree::
   :maxdepth: 1
   :caption: Python API
   
   doc/PythonAPI/Orca/orca.rst
   
.. toctree::
   :maxdepth: 1
   :caption: Real-World Application
   
   doc/Application/presentations.md
   doc/Application/powered-by.md  
