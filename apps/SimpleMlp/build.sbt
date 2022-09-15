name := "simplemlp"

version := "0.1.0-SNAPSHOT"

organization := "com.intel.analytics.bigdl.tutorial"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.4.6" % "compile",
  "org.apache.spark" % "spark-mllib_2.11" % "2.4.6" % "compile",
  "com.intel.analytics.bigdl" % "bigdl-dllib-spark_2.4.6" % "2.1.0-SNAPSHOT" changing()
)

resolvers ++= Seq(
  "Local Maven Repository" at Path.userHome.asFile.toURI.toURL + ".m2/repository",
  "ossrh repository" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

lazy val commonSettings = Seq(
  version := "0.1.0",
  organization := "com.intel.analytics.bigdl.tutorial",
  scalaVersion := "2.11.8"
)

lazy val app = (project in file(".")).
  settings(commonSettings: _*)

assemblyMergeStrategy in assembly := {
 case PathList("META-INF", xs @ _*) => MergeStrategy.discard
 case x => MergeStrategy.first
}
