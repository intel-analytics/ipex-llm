FROM ubuntu:14.04

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL

WORKDIR /opt/work

ENV JAVA_HOME                /opt/work/jdk
ENV JAVA_OPTS                -Djenkins.install.runSetupWizard=false
ENV JENKINS_HOME             /opt/work/jenkins
ENV JENKINS_SLAVE_AGENT_PORT 50000
ENV JENKINS_VERSION          2.60.3
ENV JENKINS_SHA              2d71b8f87c8417f9303a73d52901a59678ee6c0eefcf7325efed6035ff39372a
ENV JENKINS_URL              https://repo.jenkins-ci.org/public/org/jenkins-ci/main/jenkins-war/${JENKINS_VERSION}/jenkins-war-${JENKINS_VERSION}.war
ENV JENKINS_UC               https://updates.jenkins.io
ENV JENKINS_UC_EXPERIMENTAL  https://updates.jenkins.io/experimental
ENV COPY_REFERENCE_FILE_LOG  ${JENKINS_HOME}/copy_reference_file.log
ENV PATH                     ${JAVA_HOME}/bin:${PATH}

COPY init.groovy ${JENKINS_HOME}/init.groovy.d/basic-security.groovy
COPY jenkins-support ${JENKINS_HOME}/jenkins-support
COPY jenkins.sh ${JENKINS_HOME}/jenkins.sh
COPY install-plugins.sh ${JENKINS_HOME}/install-plugins.sh

RUN apt-get update && apt-get install -y git wget curl unzip && rm -rf /var/lib/apt/lists/* && \
# Install JDK 8
    wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 ${JAVA_HOME} && \
# Install Jenkins Master
    curl -fsSL ${JENKINS_URL} -o ${JENKINS_HOME}/jenkins.war && \
    echo "${JENKINS_SHA}  ${JENKINS_HOME}/jenkins.war" | sha256sum -c - && \
    echo "${JENKINS_VERSION}" >> ${JENKINS_HOME}/jenkins.install.InstallUtil.lastExecVersion && \
    chmod a+x ${JENKINS_HOME}/jenkins.sh && \
    chmod a+x ${JENKINS_HOME}/install-plugins.sh && \
    chmod a+x ${JENKINS_HOME}/jenkins-support && \
    ${JENKINS_HOME}/install-plugins.sh nodelabelparameter parameterized-trigger github email-ext jenkins-multijob-plugin leastload

ENTRYPOINT ["/opt/work/jenkins/jenkins.sh"]

