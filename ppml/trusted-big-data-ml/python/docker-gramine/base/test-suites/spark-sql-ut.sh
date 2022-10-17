#!/bin/bash

#Prepare Spark Test-jars and Test-classes
cd /opt
wget https://github.com/apache/spark/archive/refs/tags/v$SPARK_VERSION.zip
unzip -q v$SPARK_VERSION.zip
rm v$SPARK_VERSION.zip
mv /opt/spark-$SPARK_VERSION /opt/spark-source
cp -r /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/bin /opt/spark-source
cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}
mkdir /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-jars
cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-jars
wget https://repo1.maven.org/maven2/org/apache/spark/spark-core_2.12/$SPARK_VERSION/spark-core_2.12-$SPARK_VERSION-tests.jar
wget https://repo1.maven.org/maven2/org/apache/spark/spark-catalyst_2.12/$SPARK_VERSION/spark-catalyst_2.12-$SPARK_VERSION-tests.jar
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
#mkdir /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-classes
#cd /ppml/trusted-big-data-ml/work/spark-${SPARK_VERSION}/test-classes
wget https://repo1.maven.org/maven2/org/apache/spark/spark-sql_2.12/$SPARK_VERSION/spark-sql_2.12-$SPARK_VERSION-tests.jar
#jar xvf spark-sql_2.12-$SPARK_VERSION-tests.jar
#rm spark-sql_2.12-$SPARK_VERSION-tests.jar

sparkSqlSuites=("org.apache.spark.status.api.v1.sql.SqlResourceSuite" \
                "org.apache.spark.sql.UnsafeRowSuite" \
                "org.apache.spark.sql.execution.columnar.compression.IntegralDeltaSuite" \
                "org.apache.spark.sql.api.python.PythonSQLUtilsSuite")

mkdir -p /ppml/trusted-big-data-ml/logs/runtime
mkdir -p /ppml/trusted-big-data-ml/logs/reporter

cd /ppml/trusted-big-data-ml

for suite in "${sparkSqlSuites[@]}"
do
    while true
    do
        export sgx_command="/opt/jdk8/bin/java -cp '$SPARK_HOME/conf/:$SPARK_HOME/jars/*:$SPARK_HOME/test-jars/*:$SPARK_HOME/examples/jars/*' \
                                        -Xmx8g -Dspark.testing=true -Djdk.lang.Process.launchMechanism=posix_spawn -XX:MaxMetaspaceSize=256m -Dos.name='Linux' \
                                        -Dspark.test.home=/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION -Dspark.python.use.daemon=false -Dspark.python.worker.reuse=false \
                                        -Dspark.driver.host=127.0.0.1 org.scalatest.tools.Runner -s ${suite} -fF /ppml/trusted-big-data-ml/logs/reporter/${suite}.txt"
        gramine-sgx bash 2>&1 | tee /ppml/trusted-big-data-ml/logs/runtime/${suite}.log
        echo "##########$suite Test:"
        if [ -z "$(grep "All tests passed" /ppml/trusted-big-data-ml/logs/reporter/${suite}.txt)" ]
        then
            echo "failed"
        else
            echo "pass"
            break
        fi
    done
    echo "##########$suite Test Done"
done
