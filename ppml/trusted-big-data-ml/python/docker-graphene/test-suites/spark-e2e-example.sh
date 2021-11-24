status_8_scala_e2e=1

LOCAL_IP=192.168.0.112
DB_PATH=$1

if [ $status_8_scala_e2e -ne 0 ]; then
  SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
    -cp '/ppml/trusted-big-data-ml/work/data/sqlite_example/sqlite-jdbc-3.36.0.1.jar:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/data/sqlite_example/e2e.jar' \
    -Xmx2g \
    org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.driver.host=$LOCAL_IP \
    --conf spark.driver.memory=8g \
    --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/data/sqlite_example/sqlite-jdbc-3.36.0.1.jar,/ppml/trusted-big-data-ml/work/data/sqlite_example/e2e.jar \
    --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/data/sqlite_example/sqlite-jdbc-3.36.0.1.jar,/ppml/trusted-big-data-ml/work/data/sqlite_example/e2e.jar \
    --conf spark.sql.debug.maxToStringFields=100 \
    --jars /ppml/trusted-big-data-ml/work/data/sqlite_example/sqlite-jdbc-3.36.0.1.jar,/ppml/trusted-big-data-ml/work/data/sqlite_example/e2e.jar \
    --executor-memory 8g \
    --class test.SqlExample \
    /ppml/trusted-big-data-ml/work/data/sqlite_example/e2e.jar \
    $DB_PATH" 2>&1 > e2e-customer-profile-sgx.log
fi
status_8_scala_e2e=$(echo $?)

echo "#### example.8 Excepted result(e2e): INFO this is results count: XXX"
echo "---- example.8 Actual result: "
cat e2e-customer-profile-sgx.log | egrep -a 'INFO this is results count:'
