FROM intelanalytics/bigdl-k8s

# Download BigDL PPML jar with dependency jar
ADD ./download-bigdl-ppml.sh /opt/download-bigdl-ppml.sh
RUN chmod a+x /opt/download-bigdl-ppml.sh
RUN /opt/download-bigdl-ppml.sh

ADD ./entrypoint.sh /opt/entrypoint.sh
ENTRYPOINT [ "/opt/entrypoint.sh" ]
