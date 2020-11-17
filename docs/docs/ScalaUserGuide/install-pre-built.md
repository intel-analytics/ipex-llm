## **Download a pre-built library**

You can download the BigDL release and nightly build from the [Release Page](../release-download.md)

---
## **Link with a release version**

Currently, BigDL releases are hosted on maven central; here's an example to add the BigDL dependency to your own project:
```xml
<dependency>
    <groupId>com.intel.analytics.bigdl</groupId>
    <artifactId>bigdl-[SPARK_1.5|SPARK_1.6|SPARK_2.1|SPARK_2.2|SPARK_2.3]</artifactId>
    <version>${BIGDL_VERSION}</version>
</dependency>
```
Please choose the suffix according to your Spark platform.

SBT developers can use
```sbt
libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-[SPARK_1.5|SPARK_1.6|SPARK_2.1|SPARK_2.2|SPARK_2.3]" % "${BIGDL_VERSION}"
```
You can find the optional `${BIGDL_VERSION}` from the [Release Page](../release-download.md).

--- 
## **Link with a development version**

Currently, BigDL development version is hosted on [SonaType](https://oss.sonatype.org/content/groups/public/com/intel/analytics/bigdl/). 

To link your application with the latest BigDL development version, you should add some dependencies like [Linking with BigDL releases](#link-with-a-release-version), but set `${BIGDL_VERSION}` to `0.13.0-SNAPSHOT`, and add below repository to your pom.xml.

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
Note: if you use sbt on branch 0.7 and before, you should add this configuration to `build.sbt` like below.

```scala
val repo = "http://repo1.maven.org/maven2"
def mkl_native(os: String): String = {
    s"${repo}/com/intel/analytics/bigdl/native/mkl-java-${os}/0.3.0/mkl-java-${os}-0.3.0.jar"
}

def bigquant_native(os: String): String = {
    s"${repo}/com/intel/analytics/bigdl/bigquant/bigquant-java-${os}/0.3.0/bigquant-java-${os}-0.3.0.jar"

}

libraryDependencies += "com.intel.analytics.bigdl" % "bigdl-SPARK_2.3" % "0.7.0" exclude("com.intel.analytics.bigdl", "bigdl-core") exclude("com.intel.analytics.bigdl.core.dist", "all")

// Mac
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
