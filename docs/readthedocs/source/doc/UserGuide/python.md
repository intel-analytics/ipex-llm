# Python User Guide

---
### **1. Install**
- We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to prepare the Python environment as follows:

  ```bash
  conda create -n zoo python=3.7  # "zoo" is conda environment name, you can use any name you like.
  conda activate zoo
  ```

- You need to install JDK in the environment, and properly set the environment variable `JAVA_HOME`. __JDK8__ is highly recommended.

  You may take the following commands as a reference for installing [OpenJDK](https://openjdk.java.net/install/):

  ```bash
  # For Ubuntu
  sudo apt-get install openjdk-8-jre
  export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/

  # For CentOS
  su -c "yum install java-1.8.0-openjdk"
  export JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.282.b08-1.el7_9.x86_64/jre

  export PATH=$PATH:$JAVA_HOME/bin
  java -version  # Verify the version of JDK.
  ```

#### **1.1 Official Release**

You can install the latest release version of Analytics Zoo as follows:
```bash
pip install analytics-zoo
```
_**Note:** Installing Analytics Zoo will automatically install `bigdl==0.13.0`, `pyspark==2.4.6`, `conda-pack==0.3.1` and their dependencies if they haven't been detected in your conda environment._

#### **1.2 Nightly Build**

You can install the latest nightly build of Analytics Zoo as follows:

```bash
pip install --pre --upgrade analytics-zoo
```

Alternatively, you can find the list of the nightly build versions [here](https://pypi.org/project/analytics-zoo/#history), and install a specific version as follows: 

```bash
pip install analytics-zoo=version
```

_**Note:** If you are using a custom URL of Python Package Index, you may need to check whether the latest packages have been sync'ed with pypi. 
Or you can add the option `-i https://pypi.python.org/simple` when pip install to use pypi as the index-url._

---
### **2. Run**

_**Note:** Installing Analytics Zoo from pip will automatically install `pyspark`. To avoid possible conflicts, you are highly recommended to  **unset the environment variable `SPARK_HOME`**  if it exists in your environment._


#### **2.1 Interactive Shell**

You may test if the installation is successful using the interactive Python shell as follows:

* Type `python` in the command line to start a REPL.
* Try to run the example code below to verify the installation:

  ```python
  import zoo
  from zoo.orca import init_orca_context

  print(zoo.__version__)  # Verify the version of analytics-zoo.
  sc = init_orca_context()  # Initiation of analytics-zoo on the underlying cluster.
  ```

#### **2.2 Jupyter Notebook**

You can start the Jupyter notebook as you normally do using the following command and run Analytics Zoo programs directly in a Jupyter notebook:

```bash
jupyter notebook --notebook-dir=./ --ip=* --no-browser
```

#### **2.3 Python Script**

You can directly write Analytics Zoo programs in a Python file (e.g. script.py) and run in the command line as a normal Python program:

```bash
python script.py
```

---
### **3. Python Dependencies**

We recommend using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to manage your Python dependencies. Libraries installed in the current conda environment will be automatically distributed to the cluster when calling `init_orca_context`. You can also add extra dependencies as `.py`, `.zip` and `.egg` files by specifying `extra_python_lib` argument in `init_orca_context`. 

For more details, please refer to [Orca Context](../Orca/Overview/orca-context.md).

---
### **4. Compatibility**

Analytics Zoo has been tested on __Python 3.6 and 3.7__ with the following library versions:

```bash
pyspark==2.4.6
ray==1.2.0
tensorflow==1.15.0 or >2.0
pytorch>=1.5.0
torchvision>=0.6.0
horovod==0.19.2
mxnet>=1.6.0
bayesian-optimization==1.1.0
dask==2.14.0
h5py==2.10.0
numpy==1.18.1
opencv-python==4.2.0.34
pandas==1.0.3
Pillow==7.1.1
protobuf==3.12.0
psutil==5.7.0
py4j==0.10.7
redis==3.4.1
scikit-learn==0.22.2.post1
scipy==1.4.1
tensorboard==1.15.0
tensorboardX>=2.1
tensorflow-datasets==3.2.0
tensorflow-estimator==1.15.1
tensorflow-gan==2.0.0
tensorflow-hub==0.8.0
tensorflow-metadata==0.21.1
tensorflow-probability==0.7.0
Theano==1.0.4
```

---
### **5. Known Issues**

- If you meet the following error when `pip install analytics-zoo`:
```
ERROR: Could not find a version that satisfies the requirement pypandoc (from versions: none)
ERROR: No matching distribution found for pypandoc
Could not import pypandoc - required to package PySpark
Traceback (most recent call last):
  File "/root/anaconda3/lib/python3.8/site-packages/setuptools/installer.py", line 126, in fetch_build_egg
    subprocess.check_call(cmd)
  File "/root/anaconda3/lib/python3.8/subprocess.py", line 364, in check_call
    raise CalledProcessError(retcode, cmd)
subprocess.CalledProcessError: Command '['/root/anaconda3/bin/python', '-m', 'pip', '--disable-pip-version-check', 'wheel', '--no-deps', '-w', '/tmp/tmprefr87ue', '--quiet', 'pypandoc']' returned non-zero exit status 1.
```
This is actually caused by `pip install pyspark` in your Python environment. You can fix it by running `pip install pypandoc` first and then `pip install analytics-zoo`.
