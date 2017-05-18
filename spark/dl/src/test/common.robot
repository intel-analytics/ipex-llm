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
   Log To Console                Operate vertical ${verticalId} with ${operation} ...
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

BigDL Test
   [Arguments]         ${run_keyword}      ${verticals}
   @{verticalList}= 	 Split String 	     ${verticals}       separator=,
   :FOR                ${vertical}         IN                 @{verticalList}  
   \                   Operate Vertical    ${vertical}        start              running
   \                   Run KeyWord         ${run_keyword}
   [Teardown]          Stop Verticals      @{verticalList}      

Stop Verticals
   [Arguments]         @{verticalList}
   :FOR                ${vertical}         IN                @{verticalList}
   \                   Operate Vertical    ${vertical}       stop               deployed/stopped

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
   Check Verticals

Check Verticals
   :FOR                   ${vertical}           IN             @{verticals}
   \                      Status Equal          ${vertical}    deployed/stopped
   Status Equal           ${public_hdfs_vid}    running

Run Shell
   [Arguments]       ${program}
   ${rc}             ${output}=     Run and Return RC and Output    ${program}
   Log To Console                   ${output}
   Should Be Equal As Integers      ${rc}          0
