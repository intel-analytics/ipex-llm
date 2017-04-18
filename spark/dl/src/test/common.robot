*** Settings ***
Documentation   BigDL robot testing
Library         Collections
Library         RequestsLibrary
Library         String
Library         OperatingSystem

*** Keywords ***
Operate Vertical
   [Documentation]               Post operation to configuring service. Operation allowed: deploy, stop, suspend, resume, clear, reset
   [Arguments]                   ${verticalId}       ${operation}                          ${expectStatus}
   Create Session                host                http://${ardaHost}:10021
   Log To Console                Operate vertical    ${verticalId} with ${operation} ...
   ${resp}=                      Post Request        host                                  /vertical/${verticalId}/operation   data=${operation}
   ${statusCode}=                Convert To String   ${resp.status_code}
   Should Start With             ${statusCode}       20
   Wait Until Keyword Succeeds   10 min              5 sec                                 Status Equal                        ${verticalId}       ${expectStatus}

Status Equal
   [Documentation]                  Match certain vertical's status
   [Arguments]                      ${verticalId}                              ${status}
   Create Session                   host                                       http://${ardaHost}:10021
   Log To Console                   Get vertical ${verticalId}'s status ...
   ${resp}=                         Get Request                                host                        /vertical/${verticalId}
   ${statusCode}=                   Convert To String                          ${resp.status_code}
   Should Start With                ${statusCode}                              20
   ${json}=                         To Json                                    ${resp.content}
   Dictionary Should Contain Key    ${json}                                    status
   ${realStatus}=                   Get From Dictionary                        ${json}                     status
   Log To Console                   Expected=${status}, Actual=${realStatus}
   Should Be Equal As Strings       ${status}                                  ${realStatus}

BigDL Integration Test
   [Arguments]        ${verticalId}                                                      ${suite}         ${mvnarg}    ${argLine}
   Operate Vertical   ${verticalId}                                                      start            running
   ${build}= 	      Catenate                         SEPARATOR=/                       ${curDir}        make-dist.sh
   Run Keyword If     '${mvnarg}' == 'spark_2.0'       Log To Console   start to build jar for spark2.0
#   Run Keyword If     '${mvnarg}' == 'spark_2.0'       Run                               ${build} -P ${mvnarg}
#   Run Keyword If     '${mvnarg}' == 'spark_2.0'       Run                               \cp spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar ${jar_path}
   Run Keyword If     '${mvnarg}' == 'spark_2.0'       Log To Console                    build jar spark2.0 finished
   Run Keyword If     '${mvnarg}' == 'spark_2.1'       Log To Console                    start to build jar for spark2.1
   Run Keyword If     '${mvnarg}' == 'spark_2.1'       Run                               ${build} -P ${mvnarg}
   Run Keyword If     '${mvnarg}' == 'spark_2.1'       Run                               \cp spark/dl/target/bigdl-0.2.0-SNAPSHOT-jar-with-dependencies.jar ${jar_path} 
   Run Keyword If     '${mvnarg}' == 'spark_2.1'       Log To Console                    build jar spark2.1 finished
   Run                mvn clean test -Dsuites=${suite} -DargLine="${argLine}" -P integration-test -P ${mvnarg} > temp.log 2>&1
   ${stdout}=         Get File                                                           temp.log
   Log To Console     ${stdout}
   Should Contain     ${stdout}                                                          BUILD SUCCESS
   [Teardown]         Operate Vertical                                                   ${verticalId}    stop         deployed/stopped
   
BigDL Example Test
   [Arguments]        ${verticalId}       ${suite}                                                         ${argLine}
   Operate Vertical   ${verticalId}       start                                                            running
   ${result}=         Run                 ${program}
   Log To Console     ${result}           
   Should Contain     ${result}           success

Check DataSource
   Create Session   webhdfs               http://${public_hdfs_host}:50070
   ${resp}=         Get Request           webhdfs        /webhdfs/v1/${imagenet}?op=GETFILESTATUS
   Should Contain   ${resp.content}       DIRECTORY
   ${resp}=         Get Request           webhdfs        /webhdfs/v1/${mnist}?op=GETFILESTATUS
   Should Contain   ${resp.content}       DIRECTORY    
   ${resp}=         Get Request           webhdfs        /webhdfs/v1/${cifar}?op=GETFILESTATUS
   Should Contain   ${resp.content}       DIRECTORY

Prepare DataSource And Verticals
   Check DataSource
   :FOR                ${vertical}           IN             @{verticals}
   \                   Status Equal          ${vertical}    deployed/stopped
   Status Equal        ${public_hdfs_vid}    running
