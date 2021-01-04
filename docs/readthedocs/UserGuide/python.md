# Python User Guide

---

### **1. Instal**
We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the environment, especially when running on distributed cluster.

```bash
conda create -n zoo python=3.7 # "zoo" is conda enviroment name, you can use any name you like.
conda activate zoo
```
#### **1.1 Official Release**

You can install the latest release version of Analytics Zoo as follows:
```bash
pip install analytics-zoo
```
_**Note:** Installing Analytics Zoo will automatically install `bigdl==0.12.1`, `pyspark==2.4.3`, `conda-pack==0.3.1` and their dependencies._

#### **1.2 Nightly Build**

<TODO: verify nightly release using pipe --pre>

You can install the latest nightly build of Analytics Zoo as follows:

```bash
pip install --pre analytics-zoo
```

<TODO: maintain a table of nightly build version and add a link here?>

Alternatively, you can find the list of the nightly build versions [here](), and install a specific version as follows: 

```bash
pip install analytics-zoo=xxx
```

### **2. Run**

_**Note:**  Installing Analytics Zoo from pip will automatically install  `pyspark`. To avoid possible conflicts, you are highly recommended to  **unset  `SPARK_HOME`**  if it exists in your environment._

#### **2.1 Interactive Shell**

You may test if the installation was successful using the interactive Python shell as follows:

#### **2.2 Jupyter Notebook**

#### **2.3 Python Script**


### **3. Compatibility**

Analytics Zoo has been tested on Python 3.7 with the following library versions: <TODO: add the library list>
