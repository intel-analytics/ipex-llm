
## Install BigDL-0.1.1 ##

1. Download Spark2.x:  
```bash
wget https://d3kbcqa49mib13.cloudfront.net/spark-2.1.0-bin-hadoop2.7.tgz 
```

2. Extract the tar ball and set `SPARK_HOME`
```bash
 tar -zxvf spark-2.1.0-bin-hadoop2.7.tgz
 export SPARK_HOME=path to spark-2.1.0-bin-hadoop2.7
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```bash
pip install --upgrade pip
pip install BigDL==0.1.1rc0     # for Python 2.7
pip3 install BigDL==0.1.1rc0  # for Python 3.n
```

---

## Install BigDL-0.2.0-snapshot

1. Download Spark2.x:  
```bash
wget https://d3kbcqa49mib13.cloudfront.net/spark-2.1.0-bin-hadoop2.7.tgz
```
2. Extract the tar ball and set `SPARK_HOME`
```bash
tar -zxvf spark-2.1.0-bin-hadoop2.7.tgz
export SPARK_HOME=path to spark-2.1.0-bin-hadoop2.7
```
3. Install BigDL release via pip (we tested this on pip 9.0.1)
```bash
pip install --upgrade pip
pip install BigDL==0.2.0.dev2   # for Python 2.7
pip3 install BigDL==0.2.0.dev2  # for Python 3.n
```


