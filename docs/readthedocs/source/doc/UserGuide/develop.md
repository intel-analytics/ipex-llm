# Developer Guide

---

BigDL source code is available at [GitHub](https://github.com/intel-analytics/BigDL):

```bash
git clone https://github.com/intel-analytics/BigDL.git
```

By default, `git clone` will download the development version of BigDL. If you want a release version, you can use the command `git checkout` to change the specified version.


### **1. Python**

#### **1.1 Build** 

To generate a new [whl](https://pythonwheels.com/) package for pip install, you can run the following script:

```bash
cd BigDL/python/dev
bash release_default_linux_spark246.sh default true false false  # build on Spark 2.4.6 for linux
# Use release_default_linux_spark312.sh to build on Spark 3.1.2 for linux
# Use release_default_mac_spark246.sh to build on Spark 2.4.6 for mac
# Use release_default_mac_spark312.sh to build on Spark 3.1.2 for mac
```

**Arguments:**

- The first argument is the BigDL __version__ to build for. 'default' means the default version (`BigDL/python/version.txt`) for the current branch. You can also specify a different version if you wish, e.g., '0.14.0.dev1'.
- The second argument is whether to __quick build__ BigDL Scala dependencies. You need to set it to be 'true' for the first build. In later builds, if you don't make any changes in BigDL Scala, you can set it to be 'false' so that the Scala dependencies would not be re-built.
- The third argument is whether to __upload__ the packages to pypi. Set it to 'false' if you are simply developing BigDL for your own usage.
- The fourth argument is whether to add __spark suffix__ (i.e. -spark2 or -spark3) to BigDL package names. Just set this to be 'false' if you are simply developing BigDL for your own usage.
- You can also add other profiles to build the package (if any) after the fourth argument, for example '-Ddata-store-url=..'.


After running the above command, you will find a `whl` file for each submodule of BigDL and you can then directly pip install them to your local Python environment:
```bash
# Install bigdl-nano
cd BigDL/python/nano/src/dist
pip install bigdl_nano-*.whl

# Install bigdl-dllib
cd BigDL/python/dllib/src/dist
pip install bigdl_dllib-*.whl

# Install bigdl-orca, which depends on bigdl-dllib and you need to install bigdl-dllib first
cd BigDL/python/orca/src/dist
pip install bigdl_orca-*.whl

# Install bigdl-friesian, which depends on bigdl-orca and you need to install bigdl-dllib and bigdl-orca first
cd BigDL/python/friesian/src/dist
pip install bigdl_friesian-*.whl

# Install bigdl-chronos, which depends on bigdl-orca and bigdl-nano. You need to install bigdl-dllib, bigdl-orca and bigdl-nano first
cd BigDL/python/chronos/src/dist
pip install bigdl_chronos-*.whl

# Install bigdl-serving
cd BigDL/python/serving/src/dist
pip install bigdl_serving-*.whl
```

See [here](./python.md) for more instructions to run BigDL after pip install.


#### **1.2 IDE Setup**
Any IDE that support Python should be able to run BigDL. PyCharm works fine for us.

You need to do the following preparations before starting the IDE to successfully run an Analytics Zoo Python program in the IDE:

- Build BigDL; see [here](#21-build) for more instructions.
- Prepare Spark environment by either setting `SPARK_HOME` as the environment variable or pip install `pyspark`. Note that the Spark version should match the one you build BigDL on.
- Set BIGDL_CLASSPATH:
```bash
export BIGDL_CLASSPATH=analytics-zoo/dist/lib/analytics-zoo-*-jar-with-dependencies.jar
```

- Prepare BigDL Python environment by either downloading BigDL source code from [GitHub](https://github.com/intel-analytics/BigDL) or pip install `bigdl`. Note that the BigDL version should match the one you build Analytics Zoo on.
- Add `pyzoo` and `spark-analytics-zoo.conf` to `PYTHONPATH`:
```bash
export PYTHONPATH=analytics-zoo/pyzoo:analytics-zoo/dist/conf/spark-analytics-zoo.conf:$PYTHONPATH
```

The above environmental variables should be available when running or debugging code in IDE.
* In PyCharm, go to RUN -> Edit Configurations. In the "Run/Debug Configurations" panel, you can update the above environment variables in your configuration.

### **2. Scala**

#### **2.1 Build**

Maven 3 is needed to build Analytics Zoo, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```

**Build using `make-dist.sh`**

It is highly recommended that you build Analytics Zoo using the [make-dist.sh script](https://github.com/intel-analytics/analytics-zoo/blob/master/make-dist.sh) with **Java 8**.

You can build Analytics Zoo with the following commands:
```bash
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a Analytics Zoo program. The files in `dist` include:

* **dist/lib/analytics-zoo-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/analytics-zoo-VERSION-python-api.zip**: This zip package contains all Python files of Analytics Zoo.

The instructions above will build Analytics Zoo with Spark 2.4.3. To build with other spark versions, for example building analytics-zoo with spark 2.2.0, you can use `bash make-dist.sh -Dspark.version=2.2.0 -Dbigdl.artifactId=bigdl_SPARK_2.2`.  

**Build with JDK 11**

Spark starts to supports JDK 11 and Scala 2.12 at Spark 3.0. You can use `-P spark_3.x` to specify Spark3 and scala 2.12. Additionally, `make-dist.sh` default uses Java 8. To compile with Java 11, it is required to specify building opts `-Djava.version=11 -Djavac.version=11`. You can build with `make-dist.sh`.

It's recommended to download [Oracle JDK 11](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html). This will avoid possible incompatibilities with maven plugins. You should update `PATH` and make sure your `JAVA_HOME` environment variable is set to Java 11 if you're running from the command line. If you're running from an IDE, you need to make sure it is set to run maven with your current JDK. 

Build with `make-dist.sh`:
 
```bash
$ bash make-dist.sh -P spark_3.x -Djava.version=11 -Djavac.version=11
```

#### **2.2 IDE Setup**

Analytics Zoo uses maven to organize project. You should choose an IDE that supports Maven project and scala language. IntelliJ IDEA works fine for us.

In IntelliJ, you can open Analytics Zoo project root directly, and the IDE will import the project automatically.

We set the scopes of spark related libraries to `provided` in the maven pom.xml, which, however, will cause a problem in IDE  (throwing `NoClassDefFoundError` when you run applications). You can easily change the scopes using the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one".
