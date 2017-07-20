## **Download BigDL Source**

BigDL source code is available at [GitHub](https://github.com/intel-analytics/BigDL)

```bash
$ git clone https://github.com/intel-analytics/BigDL.git
```

By default, `git clone` will download the development version of BigDL, if you want a release version, you can use command `git checkout` to change the version. Available release versions is [BigDL releases](https://github.com/intel-analytics/BigDL/releases).


## **Setup Build Environment**

The following instructions are aligned with master code.

Maven 3 is needed to build BigDL, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```
When compiling with Java 7, you need to add the option “-XX:MaxPermSize=1G”. 


## **Build with script (Recommended)**

It is highly recommended that you build BigDL using the [make-dist.sh script](https://github.com/intel-analytics/BigDL/blob/master/make-dist.sh). And it will handle the MAVEN_OPTS variable.

Once downloaded, you can build BigDL with the following commands:
```bash
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a BigDL program. The files in `dist` include:

* **dist/bin/bigdl.sh**: A script used to set up proper environment variables and launch the BigDL program.
* **dist/lib/bigdl-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/bigdl-VERSION-python-api.zip**: This zip package contains all Python files of BigDL.
* **dist/conf/spark-bigdl.conf**: This file contains necessary property configurations. ```Engine.createSparkConf``` will populate these properties, so try to use that method in your code. Or you need to pass the file to Spark with the "--properties-file" option. 


## **Build for MacOS**

The instructions above will only build for Linux. To build BigDL for macOS, pass `-P mac` to the `make-dist.sh` script as follows:
```bash
$ bash make-dist.sh -P mac
```

## **Build for Windows**

To build BigDL for Windows, pass `-P win64` to the build command:
```bash
> mvn clean package -DskipTests -P win64
```

Please note that we only test it on Windows 10

* **Build for Spark 2.0 and above**

The instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10); to build for Spark 2.0 and above (which uses Scala 2.11 by default), pass `-P spark_2.x` to the `make-dist.sh` script:
```bash
$ bash make-dist.sh -P spark_2.x
```

It is highly recommended to use _**Java 8**_ when running with Spark 2.x; otherwise you may observe very poor performance.


## **Build for Scala 2.10 or 2.11**

By default, `make-dist.sh` uses Scala 2.10 for Spark 1.5.x or 1.6.x, and Scala 2.11 for Spark 2.0.x or 2.1.x. To override the default behaviors, you can pass `-P scala_2.10` or `-P scala_2.11` to `make-dist.sh` as appropriate.

---

## **Build native libs**

Note that the instructions above will skip the build of native library code, and pull the corresponding libraries from Maven Central. If you want to build the the native library code by yourself, follow the steps below:

 1.  Download and install [Intel Parallel Studio XE](https://software.intel.com//qualify-for-free-software/opensourcecontributor) in your Linux box.

 2.  Prepare build environment as follows:
 
```bash
    $ source <install-dir>/bin/compilervars.sh intel64
    $ source PATH_TO_MKL/bin/mklvars.sh intel64
```
    where the `PATH_TO_MKL` is the installation directory of the MKL.
    
 3. Full build
   
Clone BigDL as follows:
```bash
   git clone git@github.com:intel-analytics/BigDL.git --recursive 
```
For already cloned repos, just use:
```bash
   git submodule update --init --recursive 
```
If the Intel MKL is not installed to the default path `/opt/intel`, please pass your libiomp5.so's directory path to the `make-dist.sh` script:
```bash
   $ bash make-dist.sh -P full-build -DiompLibDir=<PATH_TO_LIBIOMP5_DIR> 
```
Otherwise, only pass `-P full-build` to the `make-dist.sh` script:
```bash
   $ bash make-dist.sh -P full-build
```
    
---
## **Build with Maven**

To build BigDL directly using Maven, run the command below:

```bash
$ mvn clean package -DskipTests
```
After that, you can find that the three jar packages in `PATH_To_BigDL`/target/, where `PATH_To_BigDL` is the path to the directory of the BigDL. 

Note that the instructions above will build BigDL with Spark 1.5.x or 1.6.x (using Scala 2.10) for Linux, and skip the build of native library code. Similarly, you may customize the default behaviors by passing the following parameters to maven:

 - `-P mac`: build for maxOS
 - `-P win64`: build for windows
 - `-P spark_2.x`: build for Spark 2.0 and above (using Scala 2.11). (Again, it is highly recommended to use _**Java 8**_ when running with Spark 2.0; otherwise you may observe very poor performance.)
 * `-P full-build`: full build
 * `-P scala_2.10` (or `-P scala_2.11`): build using Scala 2.10 (or Scala 2.11) 


---
## **Setup IDE**

We set the scope of spark related library to `provided` in pom.xml. The reason is that we don't want package spark related jars which will make bigdl a huge jar, and generally as bigdl is invoked by spark-submit, these dependencies will be provided by spark at run-time.

This will cause a problem in IDE. When you run applications, it will throw `NoClassDefFoundError` because the library scope is `provided`.

You can easily change the scopes by the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one". 


