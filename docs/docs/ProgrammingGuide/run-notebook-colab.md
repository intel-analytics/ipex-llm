With [Google Colaboratory](https://colab.research.google.com/), we can easily set up and run code in the cloud. This page illustrates the steps to install Analytics Zoo and run notebooks on colaboratory.

First, create or load a notebook file in colaboratory. Then, prepare the environment. You only need to install JDK and Analytics Zoo. As installing analytics-zoo from pip will automatically install pyspark, you are recommended not to install pyspark by yourself.

## **Prepare Environment**

**Install Java 8**

Run the command on the colaboratory file to install jdk 1.8:

```python
# Install jdk8
!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# Set jdk environment path which enables you to run Pyspark in your Colab environment.
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
```

**Install Analytics Zoo from pip**

You can add the following command on your colab file to install the analytics-zoo via pip easily:

```python
# Install latest release version of analytics-zoo 
# Installing analytics-zoo from pip will automatically install pyspark, bigdl, and their dependencies.
!pip install analytics-zoo
```

**Begin your code**

Call `init_nncontext()` that will create a SparkContext with optimized performance configurations.

```python
from zoo.common.nncontext import*

sc = init_nncontext()
```

Output on Colaboratory:

```
Prepending /usr/local/lib/python3.6/dist-packages/bigdl/share/conf/spark-bigdl.conf to sys.path
Adding /usr/local/lib/python3.6/dist-packages/zoo/share/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.7.0-jar-with-dependencies.jar to BIGDL_JARS
Prepending /usr/local/lib/python3.6/dist-packages/zoo/share/conf/spark-analytics-zoo.conf to sys.path
```

## **Run Github Notebook on colaboratory**

If you would like to open Analytics Zoo Notebook in a GitHub repo directly, the only thing you need to do is:

- Open the Notebook file on GitHub in a browser (So the URL ends in *.ipynb*).

- Change the URL from [https://github.com/full_path_to_the_notebook]() to [https://colab.research.google.com/github/full_path_to_the_notebook]()

  For example, change the URL of Analytics Zoo tutorial [https://github.com/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb](https://github.com/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb) 
  to [https://colab.research.google.com/github/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb](https://colab.research.google.com/github/intel-analytics/zoo-tutorials/blob/master/keras/2.1-a-first-look-at-a-neural-network.ipynb).

Then, prepare the environment of Java8 and Analytics Zoo as described [above](#prepare-environment) at the beginning of the colab notebook. If you would like to save the changes, you can make a copy to drive and run it within the instructions.
