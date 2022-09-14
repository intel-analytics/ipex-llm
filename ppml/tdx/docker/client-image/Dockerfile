FROM intelanalytics/bigdl-k8s

ADD ./spark-submit-with-ppml-tdx-local.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-local.sh
ADD ./spark-submit-with-ppml-tdx-k8s.sh /opt/spark/work-dir/ppml-tdx/spark-submit-with-ppml-tdx-k8s.sh
ADD ./pod-template.yaml /opt/spark/work-dir/ppml-tdx/pod-template.yaml
ADD ./tdx-client.yaml /opt/spark/work-dir/ppml-tdx/tdx-client.yaml

ENTRYPOINT [ "/opt/entrypoint.sh" ]
