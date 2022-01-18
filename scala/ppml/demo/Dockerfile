FROM 10.239.45.10/arda/intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:0.14.0-SNAPSHOT

ADD bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar $BIGDL_HOME/jars
ADD ppml-conf.yaml /ppml/trusted-big-data-ml
ADD data /ppml/trusted-big-data-ml/data
RUN rm $SPARK_HOME/jars/guava-14.0.1.jar
