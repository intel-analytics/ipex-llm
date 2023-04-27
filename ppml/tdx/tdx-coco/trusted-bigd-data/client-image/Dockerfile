FROM intelanalytics/bigdl-k8s

# Download BigDL-PPML jar with dependency jar
ADD ./download-bigdl-ppml.sh /opt/download-bigdl-ppml.sh
RUN chmod a+x /opt/download-bigdl-ppml.sh
RUN /opt/download-bigdl-ppml.sh

ADD ./spark-submit-with-ppml-tdx-local.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-local.sh
ADD ./spark-submit-with-ppml-tdx-k8s.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-k8s.sh
ADD ./pod-template.yaml /opt/spark/work-dir/ppml-tdx/pod-template.yaml
ADD ./tdx-client.yaml /opt/spark/work-dir/ppml-tdx/tdx-client.yaml
ADD ./runtimeclass_kata.yaml /opt/spark/work-dir/ppml-tdx/runtimeclass_kata.yaml
