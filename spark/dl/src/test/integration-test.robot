*** Settings ***
Documentation    BigDL robot testing
Resource         common.robot
Suite Setup      Check Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Integration Test

*** Variables ***
@{verticals}   ${hdfsVerticalId} 

*** Test Cases ***     VERTICALID          TESTSPEC                                         ARGLINE 
test BigDL with hdfs   ${hdfsVerticalId}   com.intel.analytics.bigdl.integration.HdfsSpec   -DhdfsHost=${hdfsHost}-DhdfsPort=${hdfsPort}          
