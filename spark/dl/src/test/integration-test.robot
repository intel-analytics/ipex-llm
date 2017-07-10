*** Settings ***
Documentation    BigDL Integration Test
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Test

*** Variables ***                                                                                                                                                                     
@{verticals}    ${spark_200_3_vid}    ${spark_210_3_vid}    ${hdfs_264_3_vid}    ${spark_tf_210_3_vid}    ${spark_tf_163_3_vid}

*** Test Cases ***   SuiteName                             VerticalId
1                    Spark2.0 Test Suite                   ${spark_200_3_vid}
2                    Spark2.1 Test Suite                   ${spark_210_3_vid}
3                    Hdfs Test Suite                       ${hdfs_264_3_vid}
4                    PySpark2.1 Test Suite                 ${spark_tf_210_3_vid}
5                    PySpark1.6 Test Suite                 ${spark_tf_163_3_vid}
6                    Yarn Test Suite                       ${hdfs_264_3_vid}

# predefined service masters:
# hdfs_264_3_master
# spark_200_3_master
# spark_210_3_master
# spark_151_3_master
# spark_163_3_master

# predefined datasource
# mnist_data_source
# cifar_data_source
# imagenet_data_source


*** Keywords ***
Build SparkJar
   [Arguments]       ${spark_version}
   ${build}=         Catenate                        SEPARATOR=/    ${curdir}    make-dist.sh
   Log To Console    ${spark_version}
   Log To Console    start to build jar
   Run               ${build} -P ${spark_version}
   Remove File       ${jar_path}
   Move File         spark/dl/target/bigdl-${version}-jar-with-dependencies.jar    ${jar_path}
   Log To Console    build jar finished

Spark2.0 Test Suite
   Build SparkJar                   spark_2.x 
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.0.0-bin-hadoop2.7  
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.0.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_200_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 84 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 336 -e 3

Spark2.1 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.1.0-bin-hadoop2.7
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_210_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 84 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 336 -e 3

Hdfs Test Suite
   Run Shell      mvn clean test -Dsuites=com.intel.analytics.bigdl.integration.HdfsSpec -DhdfsMaster=${hdfs_264_3_master} -Dmnist=${mnist_data_source} -P integration-test -DforkMode=never

Yarn Test Suite
   Build SparkJar                   spark_2.x 
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.0.0-bin-hadoop2.7  
   Set Environment Variable         http_proxy               ${http_proxy}
   Set Environment Variable         https_proxy              ${https_proxy}
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.0.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 3 --driver-memory 150g --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 120 -e 3
   Set Environment Variable         PYSPARK_DRIVER_PYTHON    /var/jenkins_home/venv/bin/python
   Set Environment Variable         PYSPARK_PYTHON           ./venv.zip/venv/bin/python      
   Run Shell                        ${submit} --master yarn --deploy-mode client --executor-memory 2g --driver-memory 2g --executor-cores 10 --num-executors 2 --properties-file ${curdir}/dist/conf/spark-bigdl.conf --jars ${jar_path} --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --archives /var/jenkins_home/venv.zip --conf spark.driver.extraClassPath=${jar_path} --conf spark.executor.extraClassPath=bigdl-${version}-jar-with-dependencies.jar ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 200
   Remove Environment Variable      http_proxy                https_proxy              PYSPARK_DRIVER_PYTHON            PYSPARK_PYTHON


PySpark2.1 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.1.0-bin-hadoop2.7
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_tf_210_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 56 --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --jars ${jar_path} --properties-file ${curdir}/dist/conf/spark-bigdl.conf ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 224
 
PySpark1.6 Test Suite
   Build SparkJar                   spark_1.6
   Set Environment Variable         SPARK_HOME     /opt/work/spark-1.6.3-bin-hadoop2.6
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-1.6.3-bin-hadoop2.6/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_tf_163_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 56 --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --jars ${jar_path} --properties-file ${curdir}/dist/conf/spark-bigdl.conf --conf spark.driver.extraClassPath=${jar_path} --conf spark.executor.extraClassPath=bigdl-${version}-jar-with-dependencies.jar ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 224
