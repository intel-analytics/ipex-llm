export CUSTOM_IMAGE_TAG=bigdl-ppmltrusted-big-data-ml-python-gramine-custom
export SGX_MEM_SIZE=memory_size_of_sgx_in_custom_image
export SGX_LOG_LEVEL=log_level_of_sgx_in_custom_image
export BIGDL_IMAGE_NAME=bigdl-ppml-trusted-big-data-ml-python-gramine-base
export BIGDL_IMAGE_VERSION=2.1.0-SNAPSHOT

if [[ "$SGX_MEM_SIZE" == "memory_size_of_sgx_in_custom_image" ]] || [[ "$SGX_LOG_LEVEL" == "log_level_of_sgx_in_custom_image" ]]
then
    echo "Please specific SGX_MEM_SIZE and SGX_LOG_LEVEL."
else
    sudo docker build \
         --build-arg BIGDL_IMAGE_NAME=${BIGDL_IMAGE_NAME} \
         --build-arg BIGDL_IMAGE_VERSION=${BIGDL_IMAGE_VERSION} \
	 --build-arg SGX_MEM_SIZE=${SGX_MEM_SIZE} \
	 --build-arg SGX_LOG_LEVEL=${SGX_LOG_LEVEL} \
         -t $CUSTOM_IMAGE_TAG -f ./Dockerfile .
fi

