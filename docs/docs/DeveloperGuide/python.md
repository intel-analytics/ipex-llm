This page gives some general instructions and tips to build and develop Analytics Zoo for Python developers.

You are very welcome to add customized functionalities to Analytics Zoo to meet your own demands. 
You are also highly encouraged to contribute to Analytics Zoo for extra features so that other community users would get benefits as well.

---
## **Download Analytics Zoo Source Code**
Analytics Zoo source code is available at [GitHub](https://github.com/intel-analytics/analytics-zoo):

```bash
git clone https://github.com/intel-analytics/analytics-zoo.git
```

By default, `git clone` will download the development version of Analytics Zoo. If you want a release version, you can use the command `git checkout` to change the specified version.


---
## **Build whl package for pip install**
If you have modified some Python code and want to newly generate the [whl](https://pythonwheels.com/) package for pip install, you can run the following script:

```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default
```

**Arguments:**

- The first argument is the __platform__ to build for. Either 'linux' or 'mac'.
- The second argument is the analytics-zoo __version__ to build for. 'default' means the default version for the current branch. You can also specify a different version if you wish, e.g., '0.6.0.dev1'.
- You can also add other profiles to build the package, especially Spark and BigDL versions.
For example, under the situation that `pyspark==2.4.3` is a dependency, you need to add profiles `-Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.x` to build Analytics Zoo for Spark 2.4.3.


After running the above command, you will find a `whl` file under the folder `analytics-zoo/pyzoo/dist/`. You can then directly pip install it to your local Python environment:
```bash
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl     # for Python 2.7
pip3 install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl    # for Python 3.5 and Python 3.6
```

See [here](../PythonUserGuide/install/#install-from-pip-for-local-usage) for more remarks related to pip install.

See [here](../PythonUserGuide/run/#run-after-pip-install) for more instructions to run analytics-zoo after pip install.


---
## **Run in IDE**
You need to do the following preparations before starting the Integrated Development Environment (IDE) to successfully run an Analytics Zoo Python program in the IDE:

- Build Analytics Zoo. See [here](../ScalaUserGuide/install/#build-with-script-recommended) for more instructions.
- Prepare Spark environment by either setting `SPARK_HOME` as the environment variable or pip install `pyspark`. Note that the Spark version should match the one you build Analytics Zoo on.
- Set BIGDL_CLASSPATH:
```bash
export BIGDL_CLASSPATH=analytics-zoo/dist/lib/analytics-zoo-*-jar-with-dependencies.jar
```

- Prepare BigDL Python environment by either downloading BigDL from [GitHub](https://github.com/intel-analytics/BigDL) or pip install `bigdl`. Note that the BigDL version should match the one you build Analytics Zoo on.
- Add `pyzoo` and `spark-analytics-zoo.conf` to `PYTHONPATH`:
```bash
export PYTHONPATH=analytics-zoo/pyzoo:analytics-zoo/dist/conf/spark-analytics-zoo.conf:$PYTHONPATH
```
If you download BigDL from [GitHub](https://github.com/intel-analytics/BigDL), you also need to add `BigDL/pyspark` to `PYTHONPATH`:
```bash
export PYTHONPATH=BigDL/pyspark:$PYTHONPATH
```
