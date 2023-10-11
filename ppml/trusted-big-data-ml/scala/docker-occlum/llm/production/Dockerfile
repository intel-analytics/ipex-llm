ARG LLM_NAME
FROM ${LLM_NAME} as llm
RUN rm -rf /opt/entrypoint.sh

ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./mount.sh /opt/mount.sh
ADD ./add_conf.sh /opt/occlum_spark/add_conf.sh
ADD ./run_spark_on_occlum_glibc.sh /opt/run_llm_on_occlum_glibc.sh

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/run_llm_on_occlum_glibc.sh && \
    chmod a+x /opt/mount.sh && \
    chmod a+x /opt/occlum_spark/add_conf.sh


ENTRYPOINT [ "/opt/entrypoint.sh" ]
