## **Install without pip**
**Remark:**

- Only __Python 3.5__ and __Python 3.6__ are supported for now.
- Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1 and >=2.2.0. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.

**Steps:**

1. [Download Spark](https://spark.apache.org/downloads.html)

2. You can download the BigDL release and nightly build from the [Release Page](../release-download.md)
  or build the BigDL package from [source](../ScalaUserGuide/install-build-src.md).

3. Install Python dependencies:
    * BigDL only depends on `Numpy` and `Six` for now.
    * For Spark standalone cluster:
        * __If you're running in cluster mode, you need to install Python dependencies on both client and each worker node.__
        * Install Numpy: 
       ```sudo apt-get install python-numpy``` (Ubuntu)
        * Install Six: 
       ```sudo apt-get install python-six``` (Ubuntu)
       
    <a name="yarn.cluster"></a>
    * For Yarn cluster:
        - You can run BigDL Python programs on YARN clusters without changes to the cluster (e.g., no need to pre-install the Python dependencies). You can first package all the required Python dependencies into a virtual environment on the local node (where you will run the spark-submit command), and then directly use spark-submit to run the BigDL Python program on the YARN cluster (using that virtual environment). Please follow the steps below: 
        * Make sure you already install such libraries(python-setuptools, python-dev, gcc, make, zip, pip) for creating virtual environment. If not, please install them first. For example, on Ubuntu, run these commands to install:
          ```
            apt-get update
            apt-get install -y python-setuptools python-dev
            apt-get install -y gcc make
            apt-get install -y zip
            easy_install pip
          ```
         * Create dependency virtualenv package
            * Under BigDL home directory, you can find ```bin/python_package.sh```. Run this script to create dependency virtual environment according to dependency descriptions in requirements.txt. You can add your own dependencies in requirements.txt. The current requirements.txt only contains dependencies for BigDL python examples and models.
            * After running this script, there will be venv.zip and venv directory generated in current directory. Use them to submit your python jobs. Please refer to [example](run-without-pip.md#yarn.example) script of submitting bigdl python job with virtual environment in Yarn cluster.
            
        __FAQ__
        
        In case you encounter the following errors when you create the environment package using the above command:
        1. virtualenv ImportError: No module named urllib3
            - Using python in anaconda to create virtualenv may cause this problem. Try using python default in your system instead of installing virtualenv in anaconda.
        2. AttributeError: 'module' object has no attribute 'sslwrap'
            - Try upgrading `gevent` with `pip install --upgrade gevent`.
        

   
