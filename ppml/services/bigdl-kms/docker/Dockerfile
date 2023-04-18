ARG BIGDL_VERSION=2.2.0
ARG SPARK_VERSION=3.1.3

# java
FROM ubuntu:20.04 as java
ARG JDK_VERSION=8u192
ARG JDK_URL

RUN apt-get update --fix-missing && \
    env DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y tzdata apt-utils wget unzip && \
    wget $JDK_URL && \
    gunzip jdk-$JDK_VERSION-linux-x64.tar.gz && \
    tar -xf jdk-$JDK_VERSION-linux-x64.tar -C /opt && \
    rm jdk-$JDK_VERSION-linux-x64.tar && \
    mv /opt/jdk* /opt/jdk$JDK_VERSION && \
    ln -s /opt/jdk$JDK_VERSION /opt/jdk

# bigdl keywhiz
FROM maven:3.6-jdk-11
ARG BIGDL_VERSION
ARG SPARK_VERSION
ENV SPARK_VERSION               ${SPARK_VERSION}
ENV BIGDL_VERSION               ${BIGDL_VERSION}
ENV BIGDL_HOME                  /bigdl-${BIGDL_VERSION}
ARG http_proxy
ARG https_proxy

RUN apt-get update && \
    apt-get install -y --no-install-recommends --no-upgrade \
    gettext vim-common default-mysql-server && \
    echo validate_password.policy=LOW >> /etc/mysql/conf.d/mysql.cnf && \
    mkdir -p /usr/src/app
WORKDIR /usr/src/app

# caching trick to speed up build, this should allow non-dynamic dependencies to be cached
RUN git clone https://github.com/analytics-zoo/keywhiz.git && \
    cd keywhiz && \
    mkdir -p /usr/src/app/api \
             /usr/src/app/cli \
             /usr/src/app/client \
             /usr/src/app/hkdf \
             /usr/src/app/model \
             /usr/src/app/server \
             /usr/src/app/testing \
             /usr/src/app/log
RUN cd keywhiz &&\
    cp *.xml /usr/src/app/ && \
    cp api/pom.xml /usr/src/app/api/ && \
    cp cli/pom.xml /usr/src/app/cli/ && \
    cp client/pom.xml /usr/src/app/client/ && \
    cp hkdf/pom.xml /usr/src/app/hkdf/ && \
    cp model/pom.xml /usr/src/app/model/ && \
    cp server/pom.xml /usr/src/app/server/ && \
    cp testing/pom.xml /usr/src/app/testing/ && \
    cp log/pom.xml /usr/src/app/log/
RUN rm /usr/share/maven/conf/settings.xml && \
    cp /usr/src/app/keywhiz/settings.xml /usr/share/maven/conf/settings.xml && \
    mvn dependency:copy-dependencies --fail-never

# copy source required for build and install
RUN cd keywhiz && \
    cp -r api/* /usr/src/app/api/ && \
    cp -r cli/* /usr/src/app/cli/ && \
    cp -r client/* /usr/src/app/client/ && \
    cp -r hkdf/* /usr/src/app/hkdf/ && \
    cp -r model/* /usr/src/app/model/ && \
    cp -r server/* /usr/src/app/server/ && \
    cp -r testing/* /usr/src/app/testing/ && \
    cp -r log/* /usr/src/app/log/
RUN service mysql start && mvn -DskipTests=true package

# Drop privs inside container
RUN useradd -ms /bin/false keywhiz && \
    mkdir /data && \
    chown keywhiz:keywhiz /data && \
    mkdir /secrets && \
    chown keywhiz:keywhiz /secrets && \
    echo 'alias keywhiz.cli="/usr/src/app/cli/target/keywhiz-cli-*-SNAPSHOT-shaded.jar --devTrustStore --url https://keywhiz-service:4444"' >> ~/.bashrc && \
    echo 'alias key.provider="java -jar server/target/keywhiz-server-*-SNAPSHOT-shaded.jar"' >> ~/.bashrc && \
    echo salt > /usr/src/app/salt && \
    cd /usr/src/app && \
    wget https://repo1.maven.org/maven2/com/intel/analytics/bigdl/bigdl-ppml-spark_${SPARK_VERSION}/${BIGDL_VERSION}/bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}-jar-with-dependencies.jar

EXPOSE 4444

VOLUME ["/data", "/secrets"]

RUN cd keywhiz && \
    cp docker/entry.sh /usr/src/app && \
    cp docker/wizard.sh /usr/src/app && \
    cp docker/keywhiz-config.tpl /usr/src/app && \
    cp frontend-keywhiz-conf.yaml /usr/src/app && \
    chmod a+x /usr/src/app/entry.sh
ADD ./start-https-frontend.sh /usr/src/app/start-https-frontend.sh
RUN chmod a+x /usr/src/app/start-https-frontend.sh && \
    chmod a+x /usr/src/app/server/target/keywhiz-server-0.10.2-SNAPSHOT-shaded.jar

ENV LC_ALL      C.UTF-8
ENV LANG        C.UTF-8
COPY --from=java /opt/jdk  /opt/jdk8
