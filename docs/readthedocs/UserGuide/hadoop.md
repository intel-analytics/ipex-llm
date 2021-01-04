# Hadoop/YARN User Guide

---

You can run Analytics Zoo programs on standard Hadoop/YARN clusters without any changes to the cluster (i.e., no need to pre-install Analytics Zoo or any Python libraries in the cluster).

### **1. Prepare Python Environment**

You need to first use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment, and then install all the needed Python libraries (including Analytics Zoo) in the conda environment:

```bash
conda create -n zoo python=3.7 # "zoo" is conda enviroment name, you can use any name you like.
conda activate zoo
pip install analytics-zoo
# use conda or pip to install all the needed Python libraries
``` 
<TODO: install JDK?>

View the [Python User Guide]() for more details.

### **2. Use `init_orca_context`**

We recommend using `init_orca_context` in your code to run on standard Hadoop/YARN clusters in [YARN client mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn). <TODO: add detailed descriptions>

### **3. Use `spark_submit`**

If you need to run in [YARN cluster mode](https://spark.apache.org/docs/latest/running-on-yarn.html#launching-spark-on-yarn), you may use `spark_submit`  as follows:  <TODO: add detailed descriptions>
