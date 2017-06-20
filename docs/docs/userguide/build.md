---
# **Build**
---

This page shows how to install and build BigDL (on both Linux and macOS), including:

* [Download](#download)
* [Linking](#linking)
    * [Linking with BigDL releases](#linking-with-bigdl-releases)
    * [Linking with development version](#linking-with-development-version)
* [Get Source Code](#get-source-code)
* [Build](#build)
    * [Build with make-dist.sh](#build-with-make-distsh)
        * [Using the `make-dist.sh` script](#build-with-make-distsh)
        * [Build for macOS](#build-for-macos)
        * [Build for Spark 2.0 and above](#build-for-spark-20-and-above)
        * [Build using Scala 2.10 or 2.11](#build-using-scala-210-or-211)
        * [Full Build](#full-build)
    * [Build with Maven](#build-with-maven)
* [IDE Settings](#ide-settings)
* [Next Steps](#next-steps)

## **Download**
You may download the BigDL release (currently v0.1.0) and nightly build from the [Release Page](/release)

## **Linking**
### Linking with BigDL releases 
Currently, BigDL releases are hosted on maven central; here's an example to add the BigDL dependency to your own project:
```xml
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl</artifactId>
    <version>${BIGDL_VERSION}</version>
</dependency>
```
SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl" % "${BIGDL_VERSION}"
```
Since currently only BigDL 0.1.0 is released, ${BIGDL_VERSION} must be set to 0.1.0 here.

*Note*: the BigDL lib default supports Spark 1.5.x and 1.6.x; if your project runs on Spark 2.0 and 2.1, use this
```xml
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl-SPARK_2.0</artifactId>
    <version>${BIGDL_VERSION}</version>
</dependency>
```

SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.0" % "${BIGDL_VERSION}"
```

If your project runs on MacOS, you should add the dependency below,
```xml
<dependency>
    <groupId>com.intel.analytics.bigdl.native</groupId>
    <artifactId>mkl-java-mac</artifactId>
    <version>${BIGDL_VERSION}</version>
    <exclusions>
        <exclusion>
            <groupId>com.intel.analytics.bigdl.native</groupId>
            <artifactId>bigdl-native</artifactId>
        </exclusion>
    </exclusions>
</dependency>
```
SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java-mac" % "${BIGDL_VERSION}" from "http://repo1.maven.org/maven2/com/intel/analytics/bigdl/native/mkl-java-mac/${BIGDL_VERSION}/mkl-java-mac-${BIGDL_VERSION}.jar"
```

### **Linking with development version**
Currently, BigDL development version is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/). 

To link your application with the latest BigDL development version, you should add some dependencies like [Linking with BigDL releases](#linking-with-bigdl-releases), but set ${BIGDL_VERSION} to 0.2.0-SNAPSHOT, and add below repository to your pom.xml.

```xml
<repository>
    <id>sonatype</id>
    <name>sonatype repository</name>
    <url>https://oss.sonatype.org/content/groups/public/</url>
    <releases>
        <enabled>true</enabled>
    </releases>
    <snapshots>
        <enabled>true</enabled>
    </snapshots>
</repository>
```

SBT developers can use
```sbt
resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"
```

## **Get Source Code**
BigDL source code is available at [GitHub](https://github.com/intel-analytics/BigDL)

```sbt
$ git clone https://github.com/intel-analytics/BigDL.git
```

By default, `git clone` will download the development version of BigDL, if you want a release version, you can use command `git checkout` to change the version. Available release versions is [BigDL releases](https://github.com/intel-analytics/BigDL/releases).

## **Build**
The following instructions are aligned with master code.

Maven 3 is needed to build BigDL, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```{r, engine='sh'}
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```
When compiling with Java 7, you need to add the option “-XX:MaxPermSize=1G”. 

### **Build with `make-dist.sh`**
It is highly recommended that you build BigDL using the [make-dist.sh script](https://github.com/intel-analytics/BigDL/blob/master/make-dist.sh). And it will handle the MAVEN_OPTS variable.

Once downloaded, you can build BigDL with the following commands:
```sbt
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a BigDL program. The files in `dist` include:
* **dist/lib/bigdl-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/bigdl-VERSION-python-api.zip**: This zip package contains all Python files of BigDL.
* **dist/conf/spark-bigdl.conf**: This file contains necessary property configurations. ```Engine.createSparkConf``` will populate these properties, so try to use that method in your code. Or you need to pass the file to Spark with the "--properties-file" option. 

#### **Build for macOS**
The instructions above will only build for Linux. To build BigDL for macOS, pass `-P mac` to the `make-dist.sh` script as follows:
```sbt
$ bash make-dist.sh -P mac
```
#### **Build for Spark 2.0 and above**
The instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10); to build for Spark 2.0 and above (which uses Scala 2.11 by default), pass `-P spark_2.x` to the `make-dist.sh` script:
```sbt
$ bash make-dist.sh -P spark_2.x
```

It is highly recommended to use _**Java 8**_ when running with Spark 2.x; otherwise you may observe very poor performance.

#### **Build using Scala 2.10 or 2.11**
By default, `make-dist.sh` uses Scala 2.10 for Spark 1.5.x or 1.6.x, and Scala 2.11 for Spark 2.0.x or 2.1.x. To override the default behaviors, you can pass `-P scala_2.10` or `-P scala_2.11` to `make-dist.sh` as appropriate.

#### **Full Build**

Note that the instructions above will skip the build of native library code, and pull the corresponding libraries from Maven Central. If you want to build the the native library code by yourself, follow the steps below:

1.  Download and install [Intel Parallel Studio XE](https://software.intel.com//qualify-for-free-software/opensourcecontributor) in your Linux box.

2.  Prepare build environment as follows:
    ```{r, engine='sh'}
    $ source <install-dir>/bin/compilervars.sh intel64
    $ source PATH_TO_MKL/bin/mklvars.sh intel64
    ```
    where the `PATH_TO_MKL` is the installation directory of the MKL.
    
3. Full build
   
   Clone BigDL as follows:
   ```{r, engine='sh'}
   git clone git@github.com:intel-analytics/BigDL.git --recursive 
   ```
   For already cloned repos, just use:
   ```{r, engine='sh'}
   git submodule update --init --recursive 
   ```
   If the Intel MKL is not installed to the default path `/opt/intel`, please pass your libiomp5.so's directory path to
   the `make-dist.sh` script:
   ```{r, engine='sh'}
   $ bash make-dist.sh -P full-build -DiompLibDir=<PATH_TO_LIBIOMP5_DIR> 
   ```
   Otherwise, only pass `-P full-build` to the `make-dist.sh` script:
   ```{r, engine='sh'}
   $ bash make-dist.sh -P full-build
   ```
    

### **Build with Maven**
To build BigDL directly using Maven, run the command below:

```sbt
$ mvn clean package -DskipTests
```
After that, you can find that the three jar packages in `PATH_To_BigDL`/target/, where `PATH_To_BigDL` is the path to the directory of the BigDL. 

Note that the instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10) for Linux, and skip the build of native library code. Similarly, you may customize the default behaviors by passing the following parameters to maven:

 * `-P mac`: build for maxOS
 * `-P spark_2.x`: build for Spark 2.0 and above (using Scala 2.11). (Again, it is highly recommended to use _**Java 8**_ when running with Spark 2.0; otherwise you may observe very poor performance.)
 * `-P full-build`: full build
 * `-P scala_2.10` (or `-P scala_2.11`): build using Scala 2.10 (or Scala 2.11) 

## **IDE Settings**
We set the scope of spark related library to `provided` in pom.xml. The reason is that we don't want package spark related jars which will make bigdl a huge jar, and generally as bigdl is invoked by spark-submit, these dependencies will be provided by spark at run-time.

This will cause a problem in IDE. When you run applications, it will throw `NoClassDefFoundError` because the library scope is `provided`.

You can easily change the scopes by the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one". 

## **Next Steps**
* To learn how to run BigDL programs (as either a local Java program or a Spark program), you can check out the [Getting Started Page](/Getting-Started).
* To learn the details of Python support in BigDL, you can check out the [Python Support Page](/Python-Support)
