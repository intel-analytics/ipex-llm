## **Download a pre-built library**

You can download the BigDL release and nightly build from the [Release Page](../release-download.md)

---
## **Link with a release version**

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
You can find the optional `${BIGDL_VERSION}` from the [Release Page](../release-download.md).

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
