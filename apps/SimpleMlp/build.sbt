name := "simplemlp"

version := "0.1.0-SNAPSHOT"

organization := "com.github.qiuxin2012"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.4.3" % "compile",
  "org.apache.spark" % "spark-mllib_2.11" % "2.4.3" % "compile",
  "com.intel.analytics.bigdl" % "bigdl-dllib-spark_2.4.6" % "0.14.0-SNANSHOT"
)

resolvers ++= Seq(
  "Local Maven Repository" at Path.userHome.asFile.toURI.toURL + ".m2/repository"
)

lazy val commonSettings = Seq(
  version := "0.1.0",
  organization := "com.github.qiuxin2012",
  scalaVersion := "2.11.8"
)

lazy val app = (project in file(".")).
  settings(commonSettings: _*)

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
{
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
}

