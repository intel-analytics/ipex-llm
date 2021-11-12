#!/bin/bash
status_1_pyspark_sql_api_DataFrame=1
status_2_pyspark_sql_api_SQLContext=1
status_3_pyspark_sql_api_UDFRegistration=0
status_4_pyspark_sql_api_GroupedData=1
status_5_pyspark_sql_api_Column=1
status_6_pyspark_sql_api_Row_and_DataFrameNaFunctions=1
status_7_pyspark_sql_api_Window=1
status_8_pyspark_sql_api_DataframeReader=1
status_9_pyspark_sql_api_DataframeWriter=0
status_10_pyspark_sql_api_HiveContext=1
status_11_pyspark_sql_api_Catalog=1
status_12_pyspark_sql_types_module=1
status_13_pyspark_sql_functions_module=1

# entry /ppml/trusted-big-data-ml dir
cd /ppml/trusted-big-data-ml

if [ $status_1_pyspark_sql_api_DataFrame -ne 0 ]; then
echo "pysaprk sql api example.1 --- DataFrame"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.sql.broadcastTimeout=3000 \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_example.py" 2>&1 | tee test-sql-dataframe-sgx.log
status_1_pyspark_sql_api_DataFrame=$(echo $?)
fi

if [ $status_2_pyspark_sql_api_SQLContext -ne 0]; then
echo "pysaprk sql api example.2 --- SQLContext"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    --conf spark.sql.broadcastTimeout=3000 \
    /ppml/trusted-big-data-ml/work/examples/sql_context_example.py" 2>&1 | tee test-sql-context-sgx.log
status_2_pyspark_sql_api_SQLContext=$(echo $?)
fi

if [ $status_3_pyspark_sql_api_UDFRegistration -ne 0]; then
echo "pysaprk sql api example.3 --- UDFRegistration"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/spark-sql_2.12-3.1.2.jar' \
  -Xmx1g \
    org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_UDFRegistration_example.py" 2>&1 | tee test-sql-UDFRegistration.log
status_3_pyspark_sql_api_UDFRegistration=$(echo $?)
fi

if [ $status_4_pyspark_sql_api_GroupedData -ne 0]; then
echo "pysaprk sql api example.4 --- GroupedData"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_groupeddata_example.py" 2>&1 | tee test-sql-groupeddata-sgx.log
status_4_pyspark_sql_api_GroupedData=$(echo $?)
fi

if [ $status_5_pyspark_sql_api_Column -ne 0]; then
echo "pysaprk sql api example.5 --- Column"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_column_example.py" 2>&1 | tee test-sql-column-sgx.log
status_5_pyspark_sql_api_Column=$(echo $?)
fi

if [ $status_6_pyspark_sql_api_Row_and_DataFrameNaFunctions -ne 0]; then
echo "pysaprk sql api example.6 --- Row_and_DataFrameNaFunctions"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_row_func_example.py" 2>&1 | tee test-sql-row-sgx.log
status_6_pyspark_sql_api_Row_and_DataFrameNaFunctions=$(echo $?)
fi

if [ $status_7_pyspark_sql_api_Window -ne 0]; then
echo "pysaprk sql api example.7 --- Window"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_window_example.py" 2>&1 | tee test-window-sgx.log
status_7_pyspark_sql_api_Window=$(echo $?)
fi

if [ $status_8_pyspark_sql_api_DataframeReader -ne 0]; then
echo "pysaprk sql api example.8 --- DataframeReader"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_dataframe_reader_example.py" 2>&1 | tee test-dataframe-reader-sgx.log
status_8_pyspark_sql_api_DataframeReader=$(echo $?)
fi

if [ $status_9_pyspark_sql_api_DataframeWriter -ne 0]; then
echo "pysaprk sql api example.9 --- DataframeWriter"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_dataframe_writer_example.py" 2>&1 | tee test-dataframe-writer-sgx.log
status_9_pyspark_sql_api_DataframeWriter=$(echo $?)
fi

if [ $status_10_pyspark_sql_api_HiveContext -ne 0]; then
echo "pysaprk sql api example.10 --- HiveContext"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_hive_context_example.py" 2>&1 | tee sql_hive_context_example-sgx.log
status_10_pyspark_sql_api_HiveContext=$(echo $?)
fi

if [ $status_11_pyspark_sql_api_Catalog -ne 0]; then
echo "pysaprk sql api example.11 --- Catalog"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_catalog_example.py" 2>&1 | tee sql_catalog_example-sgx.log
status_11_pyspark_sql_api_Catalog=$(echo $?)
fi

if [ $status_12_pyspark_sql_types_module -ne 0]; then
echo "pysaprk sql api example.12 --- types module"
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
  -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
  -Xmx1g \
  org.apache.spark.deploy.SparkSubmit \
    --master 'local[4]' \
    --conf spark.python.use.daemon=false \
    --conf spark.python.worker.reuse=false \
    /ppml/trusted-big-data-ml/work/examples/sql_types_example.py" 2>&1 | tee sql_types_example-sgx.log
status_12_pyspark_sql_types_module=$(echo $?)
fi

if [ $status_13_pyspark_sql_functions_module -ne 0]; then
echo "pysaprk sql api example.13 --- pyspark api functions"
  for f_num in {a..g}
  do
    SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
      -cp '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
      -Xmx1g \
      org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        /ppml/trusted-big-data-ml/work/examples/sql_functions_${f_num}_example" 2>&1 | tee sql_functions_${f_num}_example.log
  done
fi

echo "#### pyspark api example.1 Excepted result(DataFrame): "
echo "---- pyspark api example.1 Actual result: "
cat test-sql-dataframe-sgx.log | egrep -a 'INFO SQL Dataframe Example API finished'

echo "#### pyspark api example.2 Excepted result(SQLContext): Finish running SQL context API"
echo "---- pyspark api example.2 Actual result: "
cat  test-sql-context-sgx.log | egrep -a 'Finish running SQL context API'

echo "#### pyspark api example.3 Excepted result(UDFRegistration): Finish running SQL UDFRegistration API"
echo "---- pyspark api example.3 Actual result: "
cat test-sql-UDFRegistration.log | egrep 'Finish running SQL UDFRegistration API'

echo "#### pyspark api example.4 Excepted result(GroupedData): Finish running SQL GroupedData API"
echo "---- pyspark api example.4 Actual result: "
cat test-sql-groupeddata-sgx.log | egrep 'Finish running SQL GroupedData API'

echo "#### pyspark api example.5 Excepted result(Column): Finish running SQL Column API"
echo "---- pyspark api example.5 Actual result: "
cat test-sql-column-sgx.log | egrep 'Finish running SQL Column API'

echo "#### pyspark api example.6 Excepted result(Row_and_DataFrameNaFunctions): Finish running SQL Row_and_DataFrameNaFunctions API"
echo "---- pyspark api example.6 Actual result: "
cat test-sql-row-sgx.log  | egrep 'Finish running SQL Row_and_DataFrameNaFunctions API'

echo "#### pyspark api example.7 Excepted result(Window): Finish running Window and WindowSpec API"
echo "---- pyspark api example.7 Actual result: "
cat test-window-sgx.log | egrep 'Finish running Window and WindowSpec API'

echo "#### pyspark api example.8 Excepted result(DataframeReader): Finish running dataframe reader API"
echo "---- pyspark api example.8 Actual result: "
cat test-dataframe-reader-sgx.log | egrep 'Finish running dataframe reader API'

echo "#### pyspark api example.9 Excepted result(DataframeWriter): Finish running dataframe writer API"
echo "---- pyspark api example.9 Actual result: "
cat test-dataframe-writer-sgx.log | egrep 'Finish running dataframe writer API'

echo "#### pyspark api example.10 Excepted result(HiveContext): Finish running HiveContext API"
echo "---- pyspark api example.10 Actual result: "
cat sql_hive_context_example-sgx.log | egrep 'Finish running HiveContext API'

echo "#### pyspark api example.11 Excepted result(Catalog): Finish running Catalog API"
echo "---- pyspark api example.11 Actual result: "
cat sql_catalog_example-sgx.log | egrep 'Finish running Catalog API'

echo "#### pyspark api example.12 Excepted result(types module): Finish running types module API"
echo "---- pyspark api example.12 Actual result: "
cat sql_types_example-sgx.log | egrep 'Finish running types module API'

for f_num in {a..g}
do
  echo "#### pyspark api example.${f_num} Excepted result(functions): Finish running function_${f_num} API"
  echo "---- pyspark api example.${f_num} Actual result: "
  cat sql_functions_${f_num}_example.log | egrep -a "Finish running function_${f_num} API"
done
