# Fraud Detection
As a highly simplified demo, this notebook uses the public data set to build a fraud detection example on Apache Spark.

The notebook is developed using Scala with Analytics Zoo for Spark 2.1 and BigDL.


Download the dataset ([Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud/data))(E.g. download to /tmp/datasets/creditcard.csv)

How to run the notebook:

1. Refer to [Apache Toree](https://github.com/apache/incubator-toree/blob/master/README.md) for
how to use scala in Jupyter notebook.

An outline is:
```bash
pip install https://dist.apache.org/repos/dist/dev/incubator/toree/0.3.0-incubating-rc1/toree-pip/toree-0.3.0.tar.gz
```
Run `export SPARK_HOME=the root directory of Spark`.

To set up scala kernal, use
```
jupyter toree install --spark_opts='--master=local[*]' --user --spark_home=$SPARK_HOME
```

2. Build Analytics Zoo jar file under Spark 2.x.

You can download Analytics Zoo prebuilt release and nightly build package from [here](https://analytics-zoo.github.io/master/#release-download/) and extract it.


3. To support the training for imbalanced data set in fraud detection, some Transformers and algorithms are developed in 
https://github.com/intel-analytics/analytics-zoo/tree/legacy/pipeline/fraudDetection. We provided a pre-built jar file in this folder. Feel free to go to the source folder and run "mvn clean package" to build from source.


4. Run the following command for Spark local mode (`MASTER=local[*]`) or cluster mode. Change parameter settings as you need, e.g. `MASTER = local[physcial_core_number]`. Recommended driver memory and executor memory is 10g. 


```bash
export ANALYTICS_ZOO_HOME=the dist directory under the Analytics Zoo project that you extract from the downloaded zip package
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

SPARK_OPTS='--master=local[*] --jars ${ANALYTICS_ZOO_JAR},./fraud-1.0.1-SNAPSHOT.jar --driver-memory 10g --executor-memory 10g' 
TOREE_OPTS='--nosparkcontext' jupyter notebook
```
