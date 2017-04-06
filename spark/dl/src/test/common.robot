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
   [Arguments]        ${verticalId}                                                      ${suite}         ${argLine}
   Operate Vertical   ${verticalId}                                                      start            running
   Run                mvn test -Dsuites=${suite} -DargLine=${argLine} -P integration-test > temp.log 2>&1
   ${stdout}=         Get File                                                           temp.log
   Log To Console     ${stdout}
   Should Contain     ${stdout}                                                          BUILD SUCCESS
   [Teardown]         Operate Vertical                                                   ${verticalId}    stop          deployed/stopped
   
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
   Data Source 
   Check DataSource
   :FOR                ${vertical}           IN             @{verticals}
   \                   Status Equal          ${vertical}    deployed/stopped
tatus Equal ${public_hdfs_vid} running
