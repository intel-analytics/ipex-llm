## **NOTES**

- Pip install support __mac__ and __linux__ platform but only __Spark1.6.x__ for now.
- Pip install only support run in __local__. Might support cluster mode in the future.
- We've tested this package with __python 2.7__ and __python 3.5__


## **Install BigDL-0.1.2**

1.Download Spark1.6.3:  
```bash
wget https://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.6.tgz 
```

2.Extract the tar ball and set SPARK_HOME
```bash
tar -zxvf spark-1.6.3-bin-hadoop2.6.tgz
export SPARK_HOME=path to spark-1.6.3-bin-hadoop2.6
```

3.Install BigDL release via pip (we tested this on pip 9.0.1)
- NOTE: you might need to `sudo` if without permission for the installation
```bash
pip install --upgrade pip
pip install BigDL==0.1.2     # for Python 2.7
pip3 install BigDL==0.1.2  # for Python 3.n
```

## **Install BigDL-0.2.0-snapshot**

1.Download Spark1.6.3:  
```bash
wget https://d3kbcqa49mib13.cloudfront.net/spark-1.6.3-bin-hadoop2.6.tgz
```

2.Extract the tar ball and set SPARK_HOME
```bash
tar -zxvf spark-1.6.3-bin-hadoop2.6.tgz
export SPARK_HOME=path to spark-1.6.3-bin-hadoop2.6
```
3.Install BigDL release via pip (we tested this on pip 9.0.1)
- NOTE: you might need to `sudo` if without permission for the installation
```bash
pip install --upgrade pip
pip install BigDL==0.2.0.dev4     # for Python 2.7
pip3 install BigDL==0.2.0.dev4  # for Python 3.n
```



