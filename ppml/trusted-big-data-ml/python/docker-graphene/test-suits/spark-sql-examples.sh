#!/bin/bash
# define scala status
cd /ppml/trusted-big-data-ml

status_2_scala_sql_example=1
status_3_scala_sql_RDDRelation=1
status_4_scala_sql_SimpleTypedAggregator=1
status_5_scala_sql_UserDefinedScalar=1
status_6_scala_sql_UserDefinedTypedAggregation=1
status_7_scala_sql_UserDefinedUntypedAggregation=1

if [ $status_2_scala_sql_example -ne 0 ]; then
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.sql.SparkSQLExample \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-example-sgx.log
fi
status_2_scala_sql_example=$(echo $?)

if [ $status_3_scala_sql_RDDRelation -ne 0 ]; then
SGX=1 ./pal_loader bash -c "rm -rf pair.parquet && \
    /opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
    -Xmx10g \
    -XX:ActiveProcessorCount=24 \
    org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --class org.apache.spark.examples.sql.RDDRelation \
    /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-RDDRelation-sgx.log
fi
status_3_scala_sql_RDDRelation=$(echo $?)

if [ $status_4_scala_sql_SimpleTypedAggregator -ne 0 ]; then
SGX=1 ./pal_loader bash -c "rm -rf pair.parquet && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.sql.SimpleTypedAggregator \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-SimpleTypedAggregator-sgx.log
fi
status_4_scala_sql_SimpleTypedAggregator=$(echo $?)

if [ $status_5_scala_sql_UserDefinedScalar -ne 0 ]; then
SGX=1 ./pal_loader bash -c "rm -rf pair.parquet && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.sql.UserDefinedScalar \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-UserDefinedScalar-sgx.log
fi
status_5_scala_sql_UserDefinedScalar=$(echo $?)

if [ $status_6_scala_sql_UserDefinedTypedAggregation -ne 0 ]; then
SGX=1 ./pal_loader bash -c "rm -rf pair.parquet && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.sql.UserDefinedTypedAggregation \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-UserDefinedTypedAggregation-sgx.log
fi
status_6_scala_sql_UserDefinedTypedAggregation=$(echo $?)

if [ $status_7_scala_sql_UserDefinedUntypedAggregation -ne 0 ]; then
SGX=1 ./pal_loader bash -c "rm -rf pair.parquet && \
  /opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx10g \
  -XX:ActiveProcessorCount=24 \
  org.apache.spark.deploy.SparkSubmit \
  --master 'local[4]' \
  --class org.apache.spark.examples.sql.UserDefinedUntypedAggregation \
  /ppml/trusted-big-data-ml/work/spark-3.1.2/examples/jars/spark-examples_2.12-3.1.2.jar" 2>&1 > test-scala-spark-sql-UserDefinedUntypedAggregation-sgx.log
fi
status_7_scala_sql_UserDefinedUntypedAggregation=$(echo $?)

# echo Results
echo "example.2 status_2_scala_sql_example"
echo -e "Excepted Result: 10 \n Actual Result:"
cat test-scala-spark-sql-example-sgx.log | egrep 'Justin' | wc -l

echo "example.3 status_3_scala_sql_RDDRelation"
echo -e "Excepted Result: 2 \n Actual Result:"
cat test-scala-spark-sql-RDDRelation-sgx.log  | grep '\[25,val_25\]' | wc -l

echo "example.4 status_4_scala_sql_SimpleTypedAggregator"
echo -e "Excepted Result: |key|TypedMax(scala.Tuple2)| \n Actual Result:"
cat test-scala-spark-sql-SimpleTypedAggregator-sgx.log | egrep -a '\|key\|TypedMax\(scala.Tuple2\)\|'

echo "example.5 status_5_scala_sql_UserDefinedScalar"
echo -e "Excepted Result: | id| \n Actual Result:"
cat test-scala-spark-sql-UserDefinedScalar-sgx.log | egrep -a '\| id\|'

echo "example.6 status_6_scala_sql_UserDefinedTypedAggregation"
echo -e "Excepted Result: |average_salary| \n Actual Result:"
cat test-scala-spark-sql-UserDefinedTypedAggregation-sgx.log | egrep -a '\|average_salary\|'

echo "example.7 status_7_scala_sql_UserDefinedUntypedAggregation"
echo -e "Excepted Result: |average_salary| \n Actual Result:"
cat test-scala-spark-sql-UserDefinedUntypedAggregation-sgx.log | egrep -a '\|average_salary\|'
