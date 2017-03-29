*** Settings ***
Documentation    BigDL robot testing
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Integration Test


*** Variables ***
@{verticals}    ${hdfs_264_3_vid}    ${spark_200_3_vid}    ${spark_210_3_vid}

# predefined service masters, you can use them in ARGLINE
# hdfs_264_3_master
# spark_200_3_master
# spark_210_3_master
# spark_151_3_master
# spark_163_3_master    

*** Test Cases ***                VERTICALID           TESTSPEC                                               MVNARG       ARGLINE
test BigDL with hdfs              ${hdfs_264_3_vid}    com.intel.analytics.bigdl.integration.HdfsSpec         1            -DhdfsMaster=${hdfs_264_3_master} -Dmnist=${mnist_data_source}
spark 2.1 standalone train        ${spark_210_3_vid}   com.intel.analytics.bigdl.integration.SparkModeSpec    spark_2.1    -Dspark.master=${spark_210_3_master} -Dspark.jars=${jar_path} -Dmnist=${mnist_data_source} -Dcifar=${cifar_data_source} -Dspark.executor.cores=28 -Dspark.cores.max=84 -Dspark.driver.memory=150g -Dspark.executor.memory=150g
spark 2.0 standalone train        ${spark_200_3_vid}   com.intel.analytics.bigdl.integration.SparkModeSpec    spark_2.0    -Dspark.master=${spark_200_3_master} -Dspark.jars=${jar_path} -Dmnist=${mnist_data_source} -Dcifar=${cifar_data_source} -Dspark.executor.cores=28 -Dspark.cores.max=84 -Dspark.driver.memory=150g -Dspark.executor.memory=150g