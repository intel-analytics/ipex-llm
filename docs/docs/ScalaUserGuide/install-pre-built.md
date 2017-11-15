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

--- 
## **Link with a development version**

Currently, BigDL development version is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/). 

To link your application with the latest BigDL development version, you should add some dependencies like [Linking with BigDL releases](#link-with-a-release-version), but set `${BIGDL_VERSION}` to `0.3.0`, and add below repository to your pom.xml.

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
Note: if you use sbt on branch 0.3 and before, you should add this configuration to `build.sbt` like below.

```scala
val repo = "http://repo1.maven.org/maven2"
def mkl_native(os: String): String = {
    s"${repo}/com/intel/analytics/bigdl/native/mkl-java-${os}/0.3.0/mkl-java-${os}-0.3.0.jar"
}

def bigquant_native(os: String): String = {
    s"${repo}/com/intel/analytics/bigdl/bigquant/bigquant-java-${os}/0.3.0/bigquant-java-${os}-0.3.0.jar"

}

libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.1" % "0.3.0" exclude("com.intel.analytics.bigdl", "bigdl-core")
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java-mac" % "0.3.0" from mkl_native("mac")
libraryDependencies += "com.intel.analytics.bigdl.bigquant" % "bigquant-java-mac" % "0.3.0" from bigquant_native("mac")
```

If you want to run it on other platforms too, append below,

```scala
// Linux
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java" % "0.3.0"
libraryDependencies += "com.intel.analytics.bigdl.bigquant" % "bigquant-java" % "0.3.0"

// Windows
libraryDependencies += "com.intel.analytics.bigdl.native" % "mkl-java-win64" % "0.3.0" from mkl_native("win64")
libraryDependencies += "com.intel.analytics.bigdl.bigquant" % "bigquant-java-win64" % "0.3.0" from bigquant_native("win64")
```

If you will assemble all dependencies to a jar. You need to add merge strategy like below.


```scala
assemblyMergeStrategy in assembly := {
    case x if x.contains("com/intel/analytics/bigdl/bigquant/") => MergeStrategy.first
    case x if x.contains("com/intel/analytics/bigdl/mkl/") => MergeStrategy.first
    case x =>
      val oldStrategy = (assemblyMergeStrategy in assembly).value
      oldStrategy(x)
}
```
