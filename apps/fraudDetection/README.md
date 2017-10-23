BigDL has been used in the Fraud Detection system for one of the top payment companies. As a highly simplified
demo, this notebook uses the public data set to build a fraud detection example on Apache Spark.

The notebook is developed using Scala with Apache Spark 2.1 and BigDL 0.3. Refer to
https://github.com/intel-analytics/analytics-zoo/tree/master/pipeline/fraudDetection for the extra feature transformers.

How to run the notebook:

1. Refer to [toree home page | https://github.com/apache/incubator-toree/blob/master/README.md] for
how to support scala in Jupyter notebook.

An outline is:
```bash
pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.2.0/snapshots/dev1/toree-pip/toree-0.2.0.dev1.tar.gz
```

2. Build BigDL jar file or download the pre-built version from https://bigdl-project.github.io/master/#release-download/ 

3. To support the training for imbalanced data set in fraud detection, some Transformers and algorithms are developed in 
https://github.com/intel-analytics/analytics-zoo/tree/master/pipeline/fraudDetection.

4. Start the notebook.

```
SPARK_OPTS='--master=local[1] --jars /path/to/bigdl/jar/file,/.../analytics-zoo/pipeline/fraudDetection/fraud.jar' TOREE_OPTS='--nosparkcontext' jupyter notebook
```
