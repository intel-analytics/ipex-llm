export CUSTOM_IMAGE_NAME=bigdl-ppml-trusted-machine-learning-gramine-custom
export CUSTOM_IMAGE_TAG=2.3.0-SNAPSHOT
export MACHINE_LEARNING_BASE_IMAGE_NAME=bigdl-ppml-trusted-machine-learning-gramine-base
export MACHINE_LEARNING_BASE_IMAGE_TAG=2.3.0-SNAPSHOT
export SGX_MEM_SIZE=memory_size_of_sgx_in_custom_image
export SGX_LOG_LEVEL=log_level_of_sgx_in_custom_image
export ENABLE_DCAP_ATTESTATION=true

if [[ "$SGX_MEM_SIZE" == "memory_size_of_sgx_in_custom_image" ]] || [[ "$SGX_LOG_LEVEL" == "log_level_of_sgx_in_custom_image" ]]
then
    echo "Please specific SGX_MEM_SIZE, SGX_LOG_LEVEL."
else
    sudo docker build \
        --build-arg MACHINE_LEARNING_BASE_IMAGE_NAME=${MACHINE_LEARNING_BASE_IMAGE_NAME} \
        --build-arg MACHINE_LEARNING_BASE_IMAGE_TAG=${MACHINE_LEARNING_BASE_IMAGE_TAG} \
        --build-arg SGX_MEM_SIZE=${SGX_MEM_SIZE} \
        --build-arg SGX_LOG_LEVEL=${SGX_LOG_LEVEL} \
        --build-arg ENABLE_DCAP_ATTESTATION=${ENABLE_DCAP_ATTESTATION} \
        -t ${CUSTOM_IMAGE_NAME}:${CUSTOM_IMAGE_TAG} \
        -f ./Dockerfile .
fi
