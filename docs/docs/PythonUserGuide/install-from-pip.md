## **NOTES**

- Pip install supports __Mac__ and __Linux__ platforms but only __Spark1.6.x__ for now.
- Pip install only supports __local__ mode. Might support cluster mode in the future.
- For those who want to use BigDL in cluster mode, try to [install without pip](./install-without-pip.md)
- We've tested this package with __Python 2.7__ and __Python 3.5__. Only these two Python versions are supported for now.


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
- NOTE: You might need to add `sudo` if without permission for the installation.
```bash
pip install --upgrade pip
pip install BigDL==0.1.2     # for Python 2.7
pip3 install BigDL==0.1.2    # for Python 3.5
```

## **Install BigDL-0.2.0**

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
- NOTE: You might need to add `sudo` if without permission for the installation.
```bash
pip install --upgrade pip
pip install BigDL==0.2.0     # for Python 2.7
pip3 install BigDL==0.2.0    # for Python 3.5
```



