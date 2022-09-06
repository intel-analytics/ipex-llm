FROM intelanalytics/bigdl-k8s

ADD ./spark-submit-with-ppml-tdx-local.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-local.sh
ADD ./spark-submit-with-ppml-tdx-k8s.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-k8s.sh
ADD ./spark-executor-template.yaml /opt/spark/work-dir/ppml-tdx/spark-executor-template.yaml 

ENTRYPOINT [ "/opt/entrypoint.sh" ]
