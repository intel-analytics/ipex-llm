## **NOTES**

- Pip install supports __Mac__ and __Linux__ platforms.
- You need to install Java __>= JDK8__ before runing BigDL, which is required by __PySpark__.
- Pip install only supports __local__ mode. Might support cluster mode in the future. For those who want to use BigDL in cluster mode, try to [install without pip](./install-without-pip.md).
- We've tested this package with __Python 3.5__ and __Python 3.6__. Support for Python 2.7 has been removed due to its end of life.
- Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1 and >=2.2.0. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.

## **Install BigDL from pip**

Install BigDL release via pip (we tested this on pip 9.0.1)

**Remark:**

- You might need to add `sudo` if without permission for the installation.

-  `numpy>=1.7`, `six>=1.10.0`, `pyspark==2.4.3` and their dependencies will be automatically installed if they haven't been detected in the current Python environment. 
- Note that installing BigDL from pip will automatically install `pyspark==2.4.3`. You are highly recommended to **unset SPARK_HOME** to avoid possible conflicts.

```bash
pip3 install --upgrade pip
pip3 install BigDL        
```
