#!/bin/bash

cd /opt
wget https://github.com/apache/spark/archive/refs/tags/v$SPARK_VERSION.zip
unzip -q v$SPARK_VERSION.zip
rm v$SPARK_VERSION.zip
mv /opt/spark-$SPARK_VERSION /opt/spark-source
cp -r /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/bin /opt/spark-source
cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
mkdir /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-jars
cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-jars
wget https://repo1.maven.org/maven2/org/apache/spark/spark-core_2.12/3.1.2/spark-core_2.12-3.1.2-tests.jar
wget https://repo1.maven.org/maven2/org/apache/spark/spark-catalyst_2.12/3.1.2/spark-catalyst_2.12-3.1.2-tests.jar
wget https://repo1.maven.org/maven2/org/scalactic/scalactic_2.12/3.1.4/scalactic_2.12-3.1.4.jar
wget https://repo1.maven.org/maven2/org/scalatest/scalatest_2.12/3.1.4/scalatest_2.12-3.1.4.jar
wget https://repo1.maven.org/maven2/org/mockito/mockito-core/3.4.6/mockito-core-3.4.6.jar
wget https://repo1.maven.org/maven2/com/h2database/h2/1.4.195/h2-1.4.195.jar
wget https://repo1.maven.org/maven2/com/ibm/db2/jcc/11.5.0.0/jcc-11.5.0.0.jar
wget https://repo1.maven.org/maven2/org/apache/parquet/parquet-avro/1.10.1/parquet-avro-1.10.1.jar
wget https://repo1.maven.org/maven2/net/bytebuddy/byte-buddy/1.10.13/byte-buddy-1.10.13.jar
wget https://repo1.maven.org/maven2/org/postgresql/postgresql/42.2.6/postgresql-42.2.6.jar
wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-mockito_2.12/1.0.0-SNAP5/scalatestplus-mockito_2.12-1.0.0-SNAP5.jar
wget https://repo1.maven.org/maven2/org/scalatestplus/scalatestplus-scalacheck_2.12/3.1.0.0-RC2/scalatestplus-scalacheck_2.12-3.1.0.0-RC2.jar
mkdir /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-classes
cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-classes
wget https://repo1.maven.org/maven2/org/apache/spark/spark-sql_2.12/3.1.2/spark-sql_2.12-3.1.2-tests.jar
jar xvf spark-sql_2.12-$SPARK_VERSION-tests.jar
rm spark-sql_2.12-$SPARK_VERSION-tests.jar