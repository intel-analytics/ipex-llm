*** Settings ***
Documentation   BigDL robot testing
Library         Collections
Library         RequestsLibrary
Library         String
Library         OperatingSystem
Library         XML

*** Keywords ***
BigDL Test
   [Arguments]         ${run_keyword}
   Log To Console      Run keyword ${run_keyword}
   Run KeyWord         ${run_keyword}

Prepare DataSource And Verticals
   Get BigDL Version

Run Shell
   [Arguments]       ${program}
   ${rc}             ${output}=     Run and Return RC and Output    ${program}
   Log To Console                   ${output}
   Should Be Equal As Integers      ${rc}          0

Get BigDL Version
   ${root}=               Parse XML           pom.xml
   ${version}=            Get Element Text    ${root}    version   
   Log To Console         ${version}
   Set Global Variable    ${version}
   ${jar_path}=           Set Variable        ${jar_dir}/bigdl-${version}-jar-with-dependencies.jar
   Set Global Variable    ${jar_path}     
