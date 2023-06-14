#!/bin/bash
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION

export config_file_path="/ppml/config.yaml"
bash /ppml/make-config.sh

# Set PCCS conf
if [ "$PCCS_URL" != "" ] ; then
    echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
    echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
fi

cd /ppml || exit

if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        rm /ppml/temp_command_file || true
        bash attestation.sh
        bash temp_command_file
    fi
    /opt/jdk11/bin/java \
            -DBackends_IP=$backends_IP \
            -Dmodel_server_home=/usr/local/lib/python3.9/dist-packages \
            -cp .:/ppml/torchserve/* \
            -Xmx1g \
            -Xms1g \
            -Xss1024K \
            -XX:MetaspaceSize=64m \
            -XX:MaxMetaspaceSize=128m \
            -XX:MaxDirectMemorySize=128m \
            org.pytorch.serve.ModelServer \
            --python /usr/bin/python3 \
            -f "$config_file_path" \
            -ncs
else
    export sgx_command="/opt/jdk11/bin/java \
            -DBackends_IP=$backends_IP \
            -Dmodel_server_home=/usr/local/lib/python3.9/dist-packages \
            -cp .:/ppml/torchserve/* \
            -Xmx1g \
            -Xms1g \
            -Xss1024K \
            -XX:MetaspaceSize=64m \
            -XX:MaxMetaspaceSize=128m \
            -XX:MaxDirectMemorySize=128m \
            org.pytorch.serve.ModelServer \
            --python /usr/bin/python3 \
            -f $config_file_path \
            -ncs"
    if [ "$ATTESTATION" = "true" ]; then
          # Also consider ENCRYPTEDFSD condition
          rm /ppml/temp_command_file || true
          bash attestation.sh
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
          fi
          echo $sgx_command >>temp_command_file
          export sgx_command="bash temp_command_file"
    else
          # ATTESTATION is false
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            # ATTESTATION false, encrypted-fsd true
            rm /ppml/temp_command_file || true
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
            echo $sgx_command >>temp_command_file
            export sgx_command="bash temp_command_file"
          fi
    fi
    gramine-sgx bash 2>&1 | tee frontend-sgx.log
    rm /ppml/temp_command_file || true
fi

