#!/bin/bash

sed -i -e "s/jenkins_slave_name/$jenkins_slave_name/g" /opt/work/jenkins/slave.groovy
sed -i -e "s/jenkins_slave_label/$jenkins_slave_label/g" /opt/work/jenkins/slave.groovy
sed -i -e "s/jenkins_slave_executors/$jenkins_slave_executors/g" /opt/work/jenkins/slave.groovy

resp=$(curl -s --user "admin:admin" -d  "script=$(<${JENKINS_HOME}/slave.groovy)" "http://${jenkins_master_host}:${jenkins_master_port}/scriptText")
token=$(echo $resp|cut -f1 -d" ")
slaveName=$(echo $resp|cut -f2 -d" ")
echo "Successfully retrived info from http://${jenkins_master_host}:${jenkins_master_port}"
echo "SlaveName is $slaveName"
echo "CSRF token is $token"
$JAVA_8_HOME/bin/java -Dorg.jenkinsci.remoting.engine.Jnlpotocol3.disabled=false -cp /opt/work/jenkins/remoting-3.14.jar hudson.remoting.jnlp.Main -headless -url "http://${jenkins_master_host}:${jenkins_master_port}" $token $slaveName
