#!/bin/bash
configFile=""
core=""
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION
TEMP_CMD_PATH=/ppml/tmp/torchserve-frontend
TEMP_CMD_FILE=$TEMP_CMD_PATH/temp_command_file
mkdir -p $TEMP_CMD_PATH
while getopts ":f:c:XA" opt
do
    case $opt in
        c)
            configFile=$OPTARG
            ;;
        f)
            core=$OPTARG
            ;;
        X)
            SGX_ENABLED="true"
            ;;
        A)
            ATTESTATION="true"
            ;;
        *)
            echo "Unknown argument passed in: $opt"
            exit 1
            ;;
    esac
done

cd /ppml || exit

if [[ $SGX_ENABLED == "false" ]]; then
    if [ "$ATTESTATION" = "true" ]; then
        cd $TEMP_CMD_PATH
        rm $TEMP_CMD_FILE || true
        bash /ppml/attestation.sh
        bash $TEMP_CMD_FILE
    fi
    taskset -c "$core" /opt/jdk11/bin/java \
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
            -f "$configFile" \
            -ncs
else
    export sgx_command="/opt/jdk11/bin/java \
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
            -f $configFile \
            -ncs"
    if [ "$ATTESTATION" = "true" ]; then
          # Also consider ENCRYPTEDFSD condition
          cd $TEMP_CMD_PATH
          rm $TEMP_CMD_FILE || true
          bash /ppml/attestation.sh
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            echo "[INFO] Distributed encrypted file system is enabled"
            bash /ppml/encrypted-fsd.sh
          fi
          echo $sgx_command >>$TEMP_CMD_FILE
          export sgx_command="bash $TEMP_CMD_FILE"
    else
          # ATTESTATION is false
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            # ATTESTATION false, encrypted-fsd true
            cd $TEMP_CMD_PATH
            rm $TEMP_CMD_FILE || true
            echo "[INFO] Distributed encrypted file system is enabled"
            bash /ppml/encrypted-fsd.sh
            echo $sgx_command >>$TEMP_CMD_FILE
            export sgx_command="bash $TEMP_CMD_FILE"
          fi
    fi
    cd /ppml || exit
    taskset -c "$core" gramine-sgx bash 2>&1 | tee frontend-sgx.log
    rm $TEMP_CMD_FILE || true
fi

