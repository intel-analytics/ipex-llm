ARG LLM_NAME
FROM ${LLM_NAME} as llm
RUN rm -rf /opt/entrypoint.sh

ADD ./entrypoint.sh /opt/entrypoint.sh
ADD ./run_llm_on_occlum_glibc.sh /opt/run_llm_on_occlum_glibc.sh

RUN chmod a+x /opt/entrypoint.sh && \
    chmod a+x /opt/run_llm_on_occlum_glibc.sh


ENTRYPOINT [ "/opt/entrypoint.sh" ]
