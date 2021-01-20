This page gives some general instructions and tips to run BigDL in IDE for Python developers.

---
## **Run in IDE**
You need to do the following preparations before starting the Integrated Development Environment (IDE) to successfully run an BigDL Python program in the IDE:
- Download and Build BigDL. See [here](../ScalaUserGuide/install-build-src) for more instructions.
- Prepare Spark environment by either setting `SPARK_HOME` as the environment variable or pip install `pyspark`.
- Set BIGDL_CLASSPATH:
```bash
export BIGDL_CLASSPATH=BigDL/dist/lib/bigdl-*-jar-with-dependencies.jar
```

- Add `pyspark` and `spark-bigdl.conf` to `PYTHONPATH`:
```bash
export PYTHONPATH=BigDL/pyspark:BigDL/dist/conf/spark-bigdl.conf:$PYTHONPATH
```
