FROM ubuntu:14.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ARG JDK_VERSION=8u192
ARG JDK_URL=your_jdk_url

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

RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

#java
RUN wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

# Install Jenkins Master
RUN curl -fsSL ${JENKINS_URL} -o ${JENKINS_HOME}/jenkins.war && \
    echo "${JENKINS_SHA}  ${JENKINS_HOME}/jenkins.war" | sha256sum -c - && \
    chmod +x ${JENKINS_HOME}/jenkins.sh

ENTRYPOINT ["/opt/work/jenkins/jenkins.sh"]
