*** Settings ***
Documentation    BigDL Integration Test
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Integration Test

*** Variables ***                                                                                                                                                                     
@{verticals}    ${spark_200_3_vid}    ${spark_210_3_vid}    ${hdfs_264_3_vid}


*** Test Cases ***   SuiteName              VerticalId
1                    Spark2.0 Test Suite    ${spark_200_3_vid}
2                    Spark2.1 Test Suite    ${spark_210_3_vid}
3                    Hdfs Test Suite        ${hdfs_264_3_vid}
4                    Yarn Test Suite        ${hdfs_264_3_vid}



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
   Move File         spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar    ${jar_path}
   Log To Console    build jar finished

Spark2.0 Test Suite
   Build SparkJar                   spark_2.0 
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.0.0-bin-hadoop2.7  
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.0.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_200_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 84 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 336 -e 3

Spark2.1 Test Suite
   Build SparkJar                   spark_2.1
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.1.0-bin-hadoop2.7
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.1.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_210_3_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 150g --executor-cores 28 --total-executor-cores 84 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 336 -e 3

Hdfs Test Suite
   Run Shell      mvn clean test -Dsuites=com.intel.analytics.bigdl.integration.HdfsSpec -DhdfsMaster=${hdfs_264_3_master} -Dmnist=${mnist_data_source} -P integration-test -DforkMode=never

Yarn Test Suite
   Build SparkJar                   spark_2.0 
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.0.0-bin-hadoop2.7  
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.0.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 1 --total-executor-cores 30 --driver-memory 150g --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 120 -e 3
