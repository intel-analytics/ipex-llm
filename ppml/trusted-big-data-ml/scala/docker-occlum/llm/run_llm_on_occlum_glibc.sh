#!/bin/bash
set -x

BLUE='\033[1;34m'
NC='\033[0m'
occlum_glibc=/opt/occlum/glibc/lib
# occlum-node IP
HOST_IP=`cat /etc/hosts | grep $HOSTNAME | awk '{print $1}'`

check_sgx_dev() {
    if [ -c "/dev/sgx/enclave" ]; then
        echo "/dev/sgx/enclave is ready"
    elif [ -c "/dev/sgx_enclave" ]; then
        echo "/dev/sgx/enclave not ready, try to link to /dev/sgx_enclave"
        mkdir -p /dev/sgx
        ln -s /dev/sgx_enclave /dev/sgx/enclave
    else
        echo "both /dev/sgx/enclave /dev/sgx_enclave are not ready, please check the kernel and driver"
    fi

    if [ -c "/dev/sgx/provision" ]; then
        echo "/dev/sgx/provision is ready"
    elif [ -c "/dev/sgx_provision" ]; then
        echo "/dev/sgx/provision not ready, try to link to /dev/sgx_provision"
        mkdir -p /dev/sgx
        ln -s /dev/sgx_provision /dev/sgx/provision
    else
        echo "both /dev/sgx/provision /dev/sgx_provision are not ready, please check the kernel and driver"
    fi

    ls -al /dev/sgx
}

init_instance() {
    # check and fix sgx device
    check_sgx_dev
    # Init Occlum instance
    cd /opt
    # check if occlum_spark exists
    [[ -d occlum_spark ]] || mkdir occlum_spark
    cd occlum_spark
    occlum init
    new_json="$(jq '.resource_limits.user_space_size = "SGX_MEM_SIZE" |
        .resource_limits.max_num_of_threads = "SGX_THREAD" |
        .metadata.debuggable = "ENABLE_SGX_DEBUG" |
        .resource_limits.kernel_space_heap_size="SGX_KERNEL_HEAP" |
        .resource_limits.kernel_space_heap_max_size="SGX_KERNEL_HEAP" |
        .entry_points = [ "/usr/lib/jvm/java-8-openjdk-amd64/bin", "/bin" ] |
        .env.untrusted = [ "OMP_NUM_THREADS", "KUBECONFIG", "MALLOC_ARENA_MAX", "ATTESTATION_DEBUG", "DMLC_TRACKER_URI", "SPARK_DRIVER_URL", "SPARK_TESTING" , "_SPARK_AUTH_SECRET" ] |
        .env.default = [ "PYTHONPATH=/host/data1/FastChat", "OCCLUM=yes","PYTHONHOME=/opt/python-occlum","LD_LIBRARY_PATH=/usr/lib/jvm/java-8-openjdk-amd64/lib/server:/usr/lib/jvm/java-8-openjdk-amd64/lib:/usr/lib/jvm/java-8-openjdk-amd64/../lib:/lib","SPARK_CONF_DIR=/opt/spark/conf","SPARK_ENV_LOADED=1","PYTHONHASHSEED=0","SPARK_HOME=/opt/spark","SPARK_SCALA_VERSION=2.12","SPARK_JARS_DIR=/opt/spark/jars","LAUNCH_CLASSPATH=/bin/jars/*",""]' Occlum.json)" && \
    echo "${new_json}" > Occlum.json
    echo "SGX_MEM_SIZE ${SGX_MEM_SIZE}"

    #copy python lib and attestation lib
    copy_bom -f /opt/python-glibc.yaml --root image --include-dir /opt/occlum/etc/template
    # enable tmp hostfs
    # --conf spark.executorEnv.USING_TMP_HOSTFS=true \
    if [[ $USING_TMP_HOSTFS == "true" ]]; then
        echo "use tmp hostfs"
        mkdir ./shuffle
        edit_json="$(cat Occlum.json | jq '.mount+=[{"target": "/tmp","type": "hostfs","source": "./shuffle"}]')" && \
        echo "${edit_json}" > Occlum.json
    fi

    if [[ -z "$SGX_MEM_SIZE" ]]; then
        sed -i "s/SGX_MEM_SIZE/20GB/g" Occlum.json
    else
        sed -i "s/SGX_MEM_SIZE/${SGX_MEM_SIZE}/g" Occlum.json
    fi

    if [[ -z "$SGX_THREAD" ]]; then
        sed -i "s/\"SGX_THREAD\"/512/g" Occlum.json
    else
        sed -i "s/\"SGX_THREAD\"/${SGX_THREAD}/g" Occlum.json
    fi

    if [[ -z "$SGX_HEAP" ]]; then
        sed -i "s/SGX_HEAP/512MB/g" Occlum.json
    else
        sed -i "s/SGX_HEAP/${SGX_HEAP}/g" Occlum.json
    fi

    if [[ -z "$SGX_KERNEL_HEAP" ]]; then
        sed -i "s/SGX_KERNEL_HEAP/1GB/g" Occlum.json
    else
        sed -i "s/SGX_KERNEL_HEAP/${SGX_KERNEL_HEAP}/g" Occlum.json
    fi

    # check attestation setting
    if [ -z "$ATTESTATION" ]; then
        echo "[INFO] Attestation is disabled!"
        ATTESTATION="false"
    fi

    if [[ $ATTESTATION == "true" ]]; then
        if [[ $PCCS_URL == "" ]]; then
           echo "[ERROR] Attestation set to true but NO PCCS"
           exit 1
        else
           echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' > /etc/sgx_default_qcnl.conf
           echo 'USE_SECURE_CERT=FALSE' >> /etc/sgx_default_qcnl.conf
           cp /etc/sgx_default_qcnl.conf /opt/occlum_spark/image/etc/
           cd /root/demos/remote_attestation/dcap/
           #build .c file
           bash ./get_quote_on_ppml.sh
           cd /opt/occlum_spark
           # dir need to exit when writing quote
           mkdir -p /opt/occlum_spark/image/etc/occlum_attestation/
           #copy bom to generate quote
           copy_bom -f /root/demos/remote_attestation/dcap/dcap-ppml.yaml --root image --include-dir /opt/occlum/etc/template
        fi
    fi

    #check glic ENV MALLOC_ARENA_MAX for docker
    if [[ -z "$MALLOC_ARENA_MAX" ]]; then
        echo "No MALLOC_ARENA_MAX specified, set to 1."
        export MALLOC_ARENA_MAX=1
    fi

    # check occlum log level for docker
    if [[ -z "$ENABLE_SGX_DEBUG" ]]; then
        echo "No ENABLE_SGX_DEBUG specified, set to off."
        export ENABLE_SGX_DEBUG=false
    fi
    export OCCLUM_LOG_LEVEL=off
    if [[ -z "$SGX_LOG_LEVEL" ]]; then
        echo "No SGX_LOG_LEVEL specified, set to off."
    else
        echo "Set SGX_LOG_LEVEL to $SGX_LOG_LEVEL"
        if [[ $SGX_LOG_LEVEL == "debug" ]] || [[ $SGX_LOG_LEVEL == "trace" ]]; then
            export ENABLE_SGX_DEBUG=true
            export OCCLUM_LOG_LEVEL=$SGX_LOG_LEVEL
        fi
    fi

    sed -i "s/\"ENABLE_SGX_DEBUG\"/$ENABLE_SGX_DEBUG/g" Occlum.json
    sed -i "s/#USE_SECURE_CERT=FALSE/USE_SECURE_CERT=FALSE/g" /etc/sgx_default_qcnl.conf
}

build_spark() {
    # Copy K8s secret
    mkdir -p image/var/run/secrets/
    cp -r /var/run/secrets/* image/var/run/secrets/

    #copy libs for attest quote in occlum
    cp -f /opt/occlum_spark/image/lib/libgomp.so.1 /opt/occlum_spark/image/opt/occlum/glibc/lib --remove-destination
    cp -f /opt/occlum_spark/image/lib/libc.so /opt/occlum_spark/image/opt/occlum/glibc/lib --remove-destination
    rm image/lib/*
    cp -f /usr/lib/x86_64-linux-gnu/*sgx* /opt/occlum_spark/image/opt/occlum/glibc/lib --remove-destination
    cp -f /usr/lib/x86_64-linux-gnu/*dcap* /opt/occlum_spark/image/opt/occlum/glibc/lib --remove-destination
    cp -f /usr/lib/x86_64-linux-gnu/libcrypt.so.1 /opt/occlum_spark/image/opt/occlum/glibc/lib --remove-destination

    # add k8s config and k8s *.yaml bash
    mkdir -p /opt/occlum_spark/image/opt/k8s/
    mkdir -p /opt/occlum_spark/image/root/.kube/
    cp -rf /opt/k8s/* /opt/occlum_spark/image/opt/k8s/
    cp -rf /root/.kube/* /opt/occlum_spark/image/root/.kube/

    # copy spark and bigdl and others dependencies
    copy_bom -f /opt/spark.yaml --root image --include-dir /opt/occlum/etc/template

    # Build
    occlum build

    #before start occlum app after occlum build
    if [[ $ATTESTATION == "true" ]]; then
        if [[ $PCCS_URL == "" ]]; then
            echo "[ERROR] Attestation set to true but NO PCCS"
            exit 1
        else
            #verify ehsm service
            cd /opt/
            bash verify-attestation-service.sh
            #register application

            #get mrenclave mrsigner
            MR_ENCLAVE_temp=$(bash print_enclave_signer.sh | grep mr_enclave)
            MR_ENCLAVE_temp_arr=(${MR_ENCLAVE_temp})
            export MR_ENCLAVE=${MR_ENCLAVE_temp_arr[1]}
            MR_SIGNER_temp=$(bash print_enclave_signer.sh | grep mr_signer)
            MR_SIGNER_temp_arr=(${MR_SIGNER_temp})
            export MR_SIGNER=${MR_SIGNER_temp_arr[1]}

            #register and get policy_Id
            policy_Id_temp=$(bash register.sh | grep policy_Id)
            policy_Id_temp_arr=(${policy_Id_temp})
            export policy_Id=${policy_Id_temp_arr[1]}
        fi
        #register error
        if [[ $? -gt 0 || -z "$policy_Id" ]]; then
            echo "can not get policy_Id, register fail"
            exit 1;
        fi
    fi

    #attestation
    if [[ $ATTESTATION == "true" ]]; then
        if [[ $PCCS_URL == "" ]]; then
            echo "[ERROR] Attestation set to /root/demos/remote_attestation/dcaprue but NO PCCS"
            exit 1
        else
                #generate dcap quote
                cd /opt/occlum_spark
                occlum run /bin/dcap_c_test $REPORT_DATA
                echo "generate quote success"
                #attest quote
                occlum run /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
                            -XX:-UseCompressedOops \
                            -XX:ActiveProcessorCount=4 \
                            -Divy.home="/tmp/.ivy" \
                            -Dos.name="Linux" \
                            -cp "$SPARK_HOME/conf/:$SPARK_HOME/jars/*:/bin/jars/*" \
                            -Xmx1g com.intel.analytics.bigdl.ppml.attestation.AttestationCLI \
                            -u $ATTESTATION_URL \
                            -i $APP_ID \
                            -k $API_KEY \
                            -c $CHALLENGE \
                            -O occlum \
                            -o $policy_Id
                if [ $? -gt 0 ]; then
                    echo "attest fail, exit"
                    exit 1;
                fi
                echo "verify success"
        fi
    fi
}



id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")

arg=$1
case "$arg" in
    init)
        init_instance
        build_spark
        ;;
    initDriver)
        init_instance
        build_spark
        ;;
    initExecutor)
        # to do
        # now executor have to register again
        init_instance
        build_spark
        ;;
esac
