## **Prerequisites**

**Download and Install Java 11**

We recommend to download [Oracle JDK 11](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html). Update PATH and/or JAVA_HOME after installation. Please make sure your JAVA_HOME environment variable is set to Java 11 if you're running from the command line. Or if you're running from an IDE, you need to make sure it is set to run maven with your current JDK.

**Download and Install Maven**

Download and install [Apache Maven](https://maven.apache.org/install.html). 

## **Build BigDL source**

**Download BigDL source code**

Clone BigDL source code.
```bash
$ git clone https://github.com/intel-analytics/BigDL.git
```

**Build with script**

Jdk 11 supports minor Scala versions, you can see scala version compatibility [description](https://docs.scala-lang.org/overviews/jdk-compatibility/overview.html). BigDL supports Spark3 with Scala 2.12. We recommend to use jdk 11 with Scala 2.12 and Spark3.
Currently, our default compiling version is java 1.8. To compile with java 11, it requires to specifiy building opts `-Djava.version=11 -Djavac.version=11`. Build with following command:
```bash
$ bash make-dist.sh -P spark_3.x -Djava.version=11 -Djavac.version=11
```
After that, you can find a dist folder, which contains all the needed files to run a BigDL program. Follow this [general guide](./install-build-src.md) to see usage of each file in the folder.

**Build with Maven**
 
To build BigDL directly using Maven, run the command below:
```bash
$ mvn clean package -DskipTests -P spark_3.x -Djava.version=11 -Djavac.version=11
```

After that, you can find that the three jar packages in PATH_To_BigDL/target/, where PATH_To_BigDL is the path to the directory of the BigDL. Follow this [general guide](./install-build-src.md) to see usage of each file in the folder.
