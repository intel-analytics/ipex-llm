#!/bin/bash

set -x

echo "BIGDL_VERSION is $BIGDL_VERSION"
echo "SPARK_VERSION is $SPARK_VERSION"

rm -rf jars
mkdir jars
cd jars

# BigDL
if [[ $BIGDL_VERSION == *"SNAPSHOT"* ]]; then
  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-ppml-spark_$SPARK_VERSION-$NIGHTLY_VERSION.jar
  mv bigdl-ppml-spark_$SPARK_VERSION-$NIGHTLY_VERSION.jar bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar

  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-dllib-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-dllib-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-dllib-spark_$SPARK_VERSION-$NIGHTLY_VERSION.jar
  mv bigdl-dllib-spark_$SPARK_VERSION-$NIGHTLY_VERSION.jar bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION.jar

  NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/core/dist/all/$BIGDL_VERSION/maven-metadata.xml | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`))
  wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/core/dist/all/$BIGDL_VERSION/all-$NIGHTLY_VERSION.jar
  mv all-$NIGHTLY_VERSION.jar all-$BIGDL_VERSION.jar
else
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION.jar
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-dllib-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-dllib-spark_$SPARK_VERSION-$BIGDL_VERSION.jar
  wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/core/dist/all/$BIGDL_VERSION/all-$BIGDL_VERSION.jar
fi


# Others
wget https://repo1.maven.org/maven2/org/scala-lang/scala-compiler/2.12.10/scala-compiler-2.12.10.jar
wget https://repo1.maven.org/maven2/org/scala-lang/scala-library/2.12.10/scala-library-2.12.10.jar
wget https://repo1.maven.org/maven2/org/apache/httpcomponents/httpclient/4.5.13/httpclient-4.5.13.jar
wget https://repo1.maven.org/maven2/org/apache/httpcomponents/httpcore/4.4.13/httpcore-4.4.13.jar
wget https://repo1.maven.org/maven2/commons-logging/commons-logging/1.1.3/commons-logging-1.1.3.jar
wget https://repo1.maven.org/maven2/com/github/scopt/scopt_2.12/3.5.0/scopt_2.12-3.5.0.jar
wget https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-core/2.17.1/log4j-core-2.17.1.jar
wget https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-slf4j-impl/2.17.1/log4j-slf4j-impl-2.17.1.jar
wget https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-1.2-api/2.17.1/log4j-1.2-api-2.17.1.jar
wget https://repo1.maven.org/maven2/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar
wget https://repo1.maven.org/maven2/org/json/json/20220320/json-20220320.jar
wget https://repo1.maven.org/maven2/org/apache/logging/log4j/log4j-api/2.17.0/log4j-api-2.17.0.jar
wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-databind/2.14.1/jackson-databind-2.14.1.jar
wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-core/2.14.1/jackson-core-2.14.1.jar
wget https://repo1.maven.org/maven2/com/fasterxml/jackson/module/jackson-module-scala_2.12/2.14.1/jackson-module-scala_2.12-2.14.1.jar
wget https://repo1.maven.org/maven2/com/fasterxml/jackson/core/jackson-annotations/2.14.1/jackson-annotations-2.14.1.jar
