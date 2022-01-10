cd /opt && \
make SGX=1 DEBUG=1 && \
graphene-sgx java -cp /opt/bigdl-0.14.0-SNAPSHOT/jars/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar com.intel.analytics.bigdl.ppml.FLServer
