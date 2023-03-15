#!/bin/bash
# Acceptable arguments: attestation, bigdl-kms, ehsm, kms-utils, pccs, all

sgx() {
    echo "Detecting SGX..."
    if [ -e "/dev/sgx" ] && [ -e "/dev/sgx_enclave" ] && [ -e "/dev/sgx_provision" ]; then
        echo "SGX driver is installed."
    else
        echo "SGX driver isn't installed."
    fi

    echo "Detecting aesmd..."
    AESMDSTATUS=$(service aesmd status | grep "Active: active (running)")
    if [ -n "$AESMDSTATUS" ]; then
        echo "aesmd is installed."
    else
        echo "aesmd isn't installed."
    fi

    echo "Detecting SGX Device Plugin for Kubernetes..."
    if [ -z $(command -v kubectl) ]; then
        echo "kubectl not found."
    else
        K8SSGXPLUGIN=$(kubectl describe node | grep sgx.intel.com)
        if [ -n "$K8SSGXPLUGIN" ]; then
            echo "SGX Device Plugin for Kubernetes is installed"
        else
            echo "SGX Device Plugin for Kubernetes isn't installed"
        fi
    fi
}

attestation () {
    echo "Detecting bigdl-attestation-service..."
    ATTESTATIONSUCCESS=""
    if [ -z "$ATTESTATION_ADDRESS" ] ; then 
        ehco "ATTESTATION_ADDRESS not found, please pro vide your ATTESTATION_ADDRESS to check bigdl-attestation-service."
    else 
        ATTESTATIONSUCCESS=$(curl -s -i -k ${ATTESTATION_ADDRESS} | grep "Welcome to BigDL Remote Attestation Service")
        if [ -z "$ATTESTATIONSUCCESS" ]; then 
            echo "bigdl-attestation-service initialization failed. Unable to connect bigdl-attestation-service."
        fi
        if [ -n "$ATTESTATIONSUCCESS" ]; then 
            echo "bigdl-attestation-service initialization is successful."
        fi
    fi
}

bigdl-kms () {
    echo "Detecting bigdl-kms..." 
    BIGDLKMSSUCCESS=""
    if [ -z "$BIGDL_KMS_IP" ] ; then 
        echo "BIGDL_KMS_IP not found, please provide your BIGDL_KMS_IP to check bigdl-kms service."
    else   
        BIGDLKMSSUCCESS=$(curl -k -s "https://${BIGDL_KMS_IP}:9876/" || grep "welcome to BigDL KMS Frontend")
        if [ -z "$BIGDLKMSSUCCESS" ]; then 
            echo "bigdl-kms initialization failed. Unable to connect BigDl KMS frontend."
        fi
        if [ -n "$BIGDLKMSSUCCESS" ]; then 
            echo "bigdl-kms initialization is successful."
        fi
    fi
}

ehsm () {
    echo "Detecting ehsm..."
    EHSMSUCCESS=""
    if [ -z "$EHSM_IP" ] ; then
        echo "EHSM_IP not found, please provide your EHSM_IP to check ehsm service."
    else 
        EHSMSUCCESS=$(curl -k -G -s "https://${EHSM_IP}:9000/ehsm?Action=Enroll" || grep "\"code\":200" >/dev/null)
        if [ -z "$EHSMSUCCESS" ]; then 
            echo "ehsm initialization failed. Unable to connect EHSM at " $EHSM_IP "."
        fi
        if [ -n "$EHSMSUCCESS" ]; then 
            echo "ehsm initialization is successful."
        fi
    fi
}

kms-utils () {
    echo "Detecting kms-utils..."
    KMSUTILSSUCCESS=""
    if [ -z "$KMS_UTILS_IP" ] ; then 
        echo "KMS_UTILS_IP not found, please provide your KMS_UTILS_IP to check kms-utils service."
    else 
        KMSUTILSSUCCESS=$(curl -k -G -s "https://${KMS_UTILS_IP}:9000/ehsm?Action=Enroll" || grep "\"code\":200" >/dev/null)
        if [ -z "$KMSUTILSSUCCESS" ]; then
            echo "kms-utils initialization failed. Unable to connect kms-utils at " $KMS_UTILS_IP "."
        fi
        if [ -n "$KMSUTILSSUCCESS" ]; then
            echo "kms-utils initialization is successful."
        fi
    fi
}

pccs () {
    echo "Detecting pccs..."
    PCCSSUCCESS=""
    if [ -z "$PCCS_IP" ] || [ -z "$PCCS_PORT" ] ; then
        echo "PCCS_IP or PCCS_PORT not found, please provide your PCCS_IP & PCCS_PORT to check pccs service."
    else 
        PCCSSUCCESS=$(curl -k -G -s -i "https://${PCCS_IP}:${PCCS_PORT}/sgx/certification/v3/rootcacrl" || grep "200")
        if [ -z $"PCCSSUCCESS" ]; then 
            echo "pccs initalization failed. Unable to connect pccs at " $PCCS_IP "."
        fi
        if [ -n $"PCCSSUCCESS" ]; then 
            echo "pccs initialization is successful."
        fi
    fi
}

all=0
if [ "$#" -lt 1 ]; then
    echo "No argument passed, detecting all component statuses."
    all=$((all+1))
else
    for arg in "$@"
    do
        if [ "$arg" == all ]; then
            echo "Detecting all component statuses."
            all=$((all+1))
        fi
    done
fi

if [ "$#" -gt 6 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"SGX\",\"attestation\", \"bigdlKMS\", \"ehsm\", \"kmsUtils\", \"pccs\""
elif [ "$all" -eq 1 ]; then 
    SGX
    attestation
    bigdl-kms
    ehsm
    kms-utils
    pccs
else 
    for arg in "$@"
    do
        if [ "$arg" == SGX ]; then
            sgx
        elif [ "$arg" == attestation ]; then
            attestation
        elif [ "$arg" == bigdl-kms ]; then
            bigdl-kms
        elif [ "$arg" == ehsm ]; then
            ehsm
        elif [ "$arg" == kms-utils ]; then
            kms-utils
        elif [ "$arg" == pccs ]; then
            pccs
        else 
            echo "Acceptable arguments: \"all\", or one or more among \"attestation\", \"bigdl-kms\", \"ehsm\", \"kms-utils\", \"pccs\""
        fi
    done
fi