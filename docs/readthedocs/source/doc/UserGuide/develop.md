# Developer Guide

---

Analytics Zoo source code is available at [GitHub](https://github.com/intel-analytics/analytics-zoo):

```bash
git clone https://github.com/intel-analytics/analytics-zoo.git
```

By default, `git clone` will download the development version of Analytics Zoo. If you want a release version, you can use the command `git checkout` to change the specified version.


### **1. Python**

#### **1.1 Build** 

To generate a new [whl](https://pythonwheels.com/) package for pip install, you can run the following script:

```bash
bash analytics-zoo/pyzoo/dev/build.sh linux default false
```

**Arguments:**

- The first argument is the __platform__ to build for. Either 'linux' or 'mac'.
- The second argument is the analytics-zoo __version__ to build for. 'default' means the default version for the current branch. You can also specify a different version if you wish, e.g., '0.6.0.dev1'.
- You can also add other profiles to build the package, especially Spark and BigDL versions.
For example, under the situation that `pyspark==2.4.3` is a dependency, you need to add profiles `-Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+` to build Analytics Zoo for Spark 2.4.3.


After running the above command, you will find a `whl` file under the folder `analytics-zoo/pyzoo/dist/`. You can then directly pip install it to your local Python environment:
```bash
pip install analytics-zoo/pyzoo/dist/analytics_zoo-VERSION-py2.py3-none-PLATFORM_x86_64.whl
```

See [here](./python.md) for more instructions to run analytics-zoo after pip install.


#### **1.2 IDE Setup**
Any IDE that support python should be able to run Analytics Zoo. PyCharm works fine for us.

You need to do the following preparations before starting the IDE to successfully run an Analytics Zoo Python program in the IDE:

- Build Analytics Zoo; see [here](#21-build) for more instructions.
- Prepare Spark environment by either setting `SPARK_HOME` as the environment variable or pip install `pyspark`. Note that the Spark version should match the one you build Analytics Zoo on.
- Set BIGDL_CLASSPATH:
```bash
export BIGDL_CLASSPATH=analytics-zoo/dist/lib/analytics-zoo-*-jar-with-dependencies.jar
```

- Prepare BigDL Python environment by either downloading BigDL source code from [GitHub](https://github.com/intel-analytics/BigDL) or pip install `bigdl`. Note that the BigDL version should match the one you build Analytics Zoo on.
- Add `pyzoo` and `spark-analytics-zoo.conf` to `PYTHONPATH`:
```bash
export PYTHONPATH=analytics-zoo/pyzoo:analytics-zoo/dist/conf/spark-analytics-zoo.conf:$PYTHONPATH
```
If you download BigDL from [GitHub](https://github.com/intel-analytics/BigDL), you also need to add `BigDL/pyspark` to `PYTHONPATH`:
```bash
export PYTHONPATH=BigDL/pyspark:$PYTHONPATH
```

The above environmental variables should be available when running or debugging code in IDE.
* In PyCharm, go to RUN -> Edit Configurations. In the "Run/Debug Configurations" panel, you can update the above environment variables in your configuration.

### **2. Scala**

#### **2.1 Build**

Maven 3 is needed to build BigDL, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```

**Build using `make-dist.sh`**

It is highly recommended that you build BigDL using the [make-dist.sh script](https://github.com/intel-analytics/BigDL/blob/branch-2.0/scala/make-dist.sh) with **Java 8**.

You can build BigDL with the following commands:
```bash
$ cd scala
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a BigDL program. The files in `dist` include:

* **dist/lib/bigdl-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/bigdl-VERSION-python-api.zip**: This zip package contains all Python files of BigDL.

The instructions above will build BigDL with Spark 2.4.6. To build with other spark versions, for example building analytics-zoo with spark 2.2.0, you can use `bash make-dist.sh -Dspark.version=2.2.0`.  

**Build with JDK 11**

Spark starts to supports JDK 11 and Scala 2.12 at Spark 3.0. You can use `-P spark_3.x` to specify Spark3 and scala 2.12. Additionally, `make-dist.sh` default uses Java 8. To compile with Java 11, it is required to specify building opts `-Djava.version=11 -Djavac.version=11`. You can build with `make-dist.sh`.

It's recommended to download [Oracle JDK 11](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html). This will avoid possible incompatibilities with maven plugins. You should update `PATH` and make sure your `JAVA_HOME` environment variable is set to Java 11 if you're running from the command line. If you're running from an IDE, you need to make sure it is set to run maven with your current JDK. 

Build with `make-dist.sh`:
 
```bash
$ bash make-dist.sh -P spark_3.x -Djava.version=11 -Djavac.version=11
```

#### **2.2 IDE Setup**

BigDL uses maven to organize project. You should choose an IDE that supports Maven project and scala language. IntelliJ IDEA works fine for us.

In IntelliJ, you can open BigDL project root directly, and the IDE will import the project automatically. If not imported automatically, right click `scala/pom.xml` and choose `Add as Maven Project`.

We set the scopes of spark related libraries to `provided` in the maven pom.xml, which, however, will cause a problem in IDE  (throwing `NoClassDefFoundError` when you run applications). You can easily change the scopes using the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one".
