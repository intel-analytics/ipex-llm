FROM intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14.0-SNAPSHOT

RUN NIGHTLY_VERSION=$(echo $(echo `wget -qO - https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/maven-metadata.xml \
    | sed -n '/<value>[0-9]*\.[0-9]*\.[0-9]*-[0-9][0-9]*\.[0-9][0-9]*-[0-9][0-9]*.*value>/p' | head -n1 | awk -F'>' '{print $2}' | tr '</value' ' '`)) && \
    wget https://oss.sonatype.org/content/repositories/snapshots/com/intel/analytics/bigdl/bigdl-ppml-spark_$SPARK_VERSION/$BIGDL_VERSION/bigdl-ppml-spark_$SPARK_VERSION-$NIGHTLY_VERSION-jar-with-dependencies.jar -O $BIGDL_HOME/jars/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar

ADD ppml-conf.yaml /ppml/trusted-big-data-ml
ADD runFlServer.sh /ppml/trusted-big-data-ml
ADD runHflClient1.sh /ppml/trusted-big-data-ml
ADD runHflClient2.sh /ppml/trusted-big-data-ml
ADD runVflClient1.sh /ppml/trusted-big-data-ml
ADD runVflClient2.sh /ppml/trusted-big-data-ml

RUN rm $SPARK_HOME/jars/guava-14.0.1.jar && \
    chmod a+x runFlServer.sh && \
    chmod a+x runHflClient1.sh && \
    chmod a+x runHflClient2.sh && \
    chmod a+x runVflClient1.sh && \
    chmod a+x runVflClient2.sh
