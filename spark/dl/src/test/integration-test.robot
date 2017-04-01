*** Settings ***
Documentation    BigDL robot testing
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Integration Test


# predefined data source, you can use them in ARGLINE
*** Keywords ***
Data Source
    ${cifar_data_source}=       Catenate    SEPARATOR=   ${public_hdfs_master}    ${cifar}
    ${mnist_data_source}=       Catenate    SEPARATOR=   ${public_hdfs_master}    ${mnist}
    ${imagenet_data_source}=    Catenate    SEPARATOR=   ${public_hdfs_master}    ${imagenet}

# predefined service masters, you can use them in ARGLINE
# hdfs_264_3_master
# spark_200_3_master
# spark_210_3_master
# spark_151_3_master
# spark_163_3_master    

*** Variables ***
@{verticals}    ${hdfs_264_3_vid}    ${spark_200_3_vid}


*** Test Cases ***      VERTICALID           TESTSPEC                                          ARGLINE 
test BigDL with hdfs    ${hdfs_264_3_vid}    com.intel.analytics.bigdl.integration.HdfsSpec    -DhdfsMaster=${hdfs_264_3_master}
