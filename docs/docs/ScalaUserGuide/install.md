## **Download a pre-built library**

You can download the Analytics Zoo release and nightly build from the [Release Page](../release-download.md)

---
## **Link with a release version**

Currently, Analytics Zoo releases are hosted on maven central; here's an example to add the Analytics Zoo dependency to your own project:
```xml
<dependency>
    <groupId>com.intel.analytics.zoo</groupId>
    <artifactId>analytics-zoo-bigdl_0.9.0-[spark_2.1.1|spark_2.2.0|spark_2.3.1|spark_2.4.3]</artifactId>
    <version>${ANALYTICS_ZOO_VERSION}</version>
</dependency>
```

SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.zoo" % "analytics-zoo-bigdl_0.9.0-[spark_2.1.1|spark_2.2.0|spark_2.3.1|spark_2.4.3]" % "${ANALYTICS_ZOO_VERSION}"
```

Remarks:

- Please choose the available suffix above according to your Spark platform and the BigDL version you want to use.
- You don't need to add the BigDL dependency to your project as it has already been packaged within Analytics Zoo.
- You can find the option `${ANALYTICS_ZOO_VERSION}` from the [Release Page](../release-download.md).

---
## **Link with a development version**

Currently, Analytics Zoo development version is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/zoo/).

To link your application with the latest Analytics Zoo development version, you should add some dependencies like [Linking with Analytics Zoo releases](#link-with-a-release-version), but set `${ANALYTICS_ZOO_VERSION}` to latest version, and add below repository to your pom.xml.

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
resolvers += "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
```

## **Download Analytics Zoo Source**

Analytics Zoo source code is available at [GitHub](https://github.com/intel-analytics/analytics-zoo)

```bash
$ git clone https://github.com/intel-analytics/analytics-zoo.git
```

By default, `git clone` will download the development version of Analytics Zoo, if you want a release version, you can use command `git checkout` to change the version.


## **Setup Build Environment**

The following instructions are aligned with master code.

Maven 3 is needed to build Analytics Zoo, you can download it from the [maven website](https://maven.apache.org/download.cgi).

After installing Maven 3, please set the environment variable MAVEN_OPTS as follows:
```bash
$ export MAVEN_OPTS="-Xmx2g -XX:ReservedCodeCacheSize=512m"
```
When compiling with Java 7, you need to add the option “-XX:MaxPermSize=1G”.


## **Build with script (Recommended)**

It is highly recommended that you build Analytics Zoo using the [make-dist.sh script](https://github.com/intel-analytics/analytics-zoo/blob/master/make-dist.sh). And it will handle the MAVEN_OPTS variable.

Once downloaded, you can build Analytics Zoo with the following commands:
```bash
$ bash make-dist.sh
```
After that, you can find a `dist` folder, which contains all the needed files to run a Analytics Zoo program. The files in `dist` include:

* **dist/lib/analytics-zoo-VERSION-jar-with-dependencies.jar**: This jar package contains all dependencies except Spark classes.
* **dist/lib/analytics-zoo-VERSION-python-api.zip**: This zip package contains all Python files of Analytics Zoo.

The instructions above will build Analytics Zoo with Spark 2.0(using Scala 2.11). It is highly recommended to use _**Java 8**_ when running with Spark 2.x; otherwise you may observe very poor performance.

## **Build for Spark 1.6**

To build for Spark 1.6(which uses Scala 2.10 by default), pass `-P spark_1.6` to the `make-dist.sh` script:
```bash
$ bash make-dist.sh -P spark_1.6
```

## **Build for Scala 2.10 or 2.11**

By default, `make-dist.sh` uses Scala 2.11 for Spark 2.1, and Scala 2.10 for Spark 1.6. To override the default behaviors, you can pass `-P scala_2.10` or `-P scala_2.11` to `make-dist.sh` as appropriate.

---
## **Build with Maven**

To build Analytics Zoo directly using Maven, run the command below:

```bash
$ mvn clean package -DskipTests
```
After that, you can find that jar packages in `PATH_TO_ANALYTICS_ZOO`/target/, where `PATH_TO_ANALYTICS_ZOO` is the path to the directory of the Analytics Zoo.

Note that the instructions above will build Analytics Zoo with Spark 2.0 (using Scala 2.11) for Linux. Similarly, you may customize the default behaviors by passing the following parameters to maven:

 - `-P spark_1.6`: build for Spark 1.6 (using Scala 2.10).
 * `-P scala_2.10` (or `-P scala_2.11`): build using Scala 2.10 (or Scala 2.11)


---
## **Setup IDE**

We set the scope of spark related library to `provided` in pom.xml. The reason is that we don't want package spark related jars which will make analytics zoo a huge jar, and generally as analytics zoo is invoked by spark-submit, these dependencies will be provided by spark at run-time.

This will cause a problem in IDE. When you run applications, it will throw `NoClassDefFoundError` because the library scope is `provided`.

You can easily change the scopes by the `all-in-one` profile.

* In Intellij, go to View -> Tools Windows -> Maven Projects. Then in the Maven Projects panel, Profiles -> click "all-in-one".
