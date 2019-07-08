## **NOTES**

- Pip install supports __Mac__ and __Linux__ platforms.
- You need to install Java __>= JDK8__ before runing BigDL, which is required by __PySpark 2.2.0__.
- Pip install only supports __local__ mode. Might support cluster mode in the future. For those who want to use BigDL in cluster mode, try to [install without pip](./install-without-pip.md).
- We've tested this package with __Python 2.7__, __Python 3.5__ and __Python 3.6__. Only these three Python versions are supported for now.
- Note that __Python 3.6__ is only compatible with Spark 1.6.4, 2.0.3, 2.1.1 and 2.2.0. See [this issue](https://issues.apache.org/jira/browse/SPARK-19019) for more discussion.


## **Install BigDL-0.9.0.dev0**

Install BigDL release via pip (we tested this on pip 9.0.1)

**Remark:**

- You might need to add `sudo` if without permission for the installation.

- `pyspark` will be automatically installed first before installing BigDL if it hasn't been detected locally.
```bash
pip install --upgrade pip
pip install BigDL==0.9.0.dev0     # for Python 2.7
pip3 install BigDL==0.9.0.dev0    # for Python 3.5 and Python 3.6
```
