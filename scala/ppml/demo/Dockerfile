FROM intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT

RUN NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml \
    | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`)) && \
    wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-ppml-spark_$SPARK_VERSION-$NIGHTLY_VERSION-jar-with-dependencies.jar -O $BIGDL_HOME/jars/bigdl-ppml-spark_$SPARK_VERSION-$BIGDL_VERSION-jar-with-dependencies.jar

ADD ppml-conf.yaml /ppml/trusted-big-data-ml
ADD ./data/* /ppml/trusted-big-data-ml/work/data/
ADD ./scripts/* /ppml/trusted-big-data-ml/

RUN rm $SPARK_HOME/jars/guava-14.0.1.jar && \
    chmod a+x runFlServer.sh && \
    chmod a+x runClient.sh
