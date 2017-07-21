
## **Install without pip**
1. [Install Spark](https://spark.apache.org/downloads.html)

2. You can download the BigDL release and nightly build from the [Release Page](../release-download.md)
  or build the BigDL package from [source](../UserGuide/install-build-src.md). 

3. Install python dependencies:
    * BigDL only depend on `Numpy` for now.  
    * For Spark standalone cluster:
        * __if you're running in cluster mode, you need to install python dependencies on both client and each worker nodes__
        * Install Numpy: 
       ```sudo apt-get install python-numpy ``` (Ubuntu)
    * For Yarn cluster:
        - You can run BigDL Python programs on YARN clusters without changes to the cluster (e.g., no need to pre-install the Python dependencies). You  can first package all the required Python dependency into a virtual environment on the localnode (where you will run the spark-submit command), and then directly use spark-submit to run the BigDL Python program on the YARN cluster (using that virtual environment). Please refer to this [Packing-dependencies](https://github.com/intel-analytics/BigDL/tree/master/pyspark/python_package) for more details.
   
