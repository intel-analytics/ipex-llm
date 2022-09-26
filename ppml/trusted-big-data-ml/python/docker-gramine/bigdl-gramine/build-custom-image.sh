export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export BIGDL_IMAGE_NAME=your_bigdl_base_image_name_used_to_build_custom_image
export BIGDL_IMAGE_VERSION=your_bigdl_base_image_version_used_to_build_customer_image
export CUSTOM_IMAGE_TAG=your_custom_image_tag
export LOCAL_IP=your_local_IP
export SGX_MEM_SIZE=memory_size_of_sgx_in_custom_image
export SGX_LOG_LEVEL=log_level_of_sgx_in_custom_image

if [[ "$BIGDL_IMAGE_NAME" == "your_bigdl_base_image_name_used_to_build_custom_image" ]] || [[ "$BIGDL_IMAGE_VERSION" == "your_bigdl_base_image_version_used_to_build_custom_image" ]] || [[ "$CUSTOM_IMAGE_TAG" == "your_custom_image_tag" ]]
then
    echo "Please specific name and version of your bigdl base name, tag of your user image to use."
else
    Proxy_Modified="sudo docker build \
            --build-arg BIGDL_IMAGE_NAME=${BIGDL_IMAGE_NAME} \
            --build-arg BIGDL_IMAGE_VERSION=${BIGDL_IMAGE_VERSION} \
            --build-arg LOCAL_IP=${LOCAL_IP} \
	    --build-arg http_proxy=http://${HTTP_PROXY_HOST}:${HTTP_PROXY_PORT} \
            --build-arg https_proxy=http://${HTTPS_PROXY_HOST}:${HTTPS_PROXY_PORT} \
	    --build-arg SGX_MEM_SIZE=${SGX_MEM_SIZE} \
	    --build-arg SGX_LOG_LEVEL=${SGX_LOG_LEVEL} \
            -t $CUSTOM_IMAGE_TAG -f ./Dockerfile ."
    
    No_Proxy_Modified="sudo docker build \
	    --build-arg BIGDL_IMAGE_NAME=${BIGDL_IMAGE_NAME} \
	    --build-arg BIGDL_IMAGE_VERSION=${BIGDL_IMAGE_VERSION} \
            --build-arg LOCAL_IP=${LOCAL_IP} \
	    --build-arg SGX_MEM_SIZE=${SGX_MEM_SIZE} \
            --build-arg SGX_LOG_LEVEL=${SGX_LOG_LEVEL} \
            -t $CUSTOM_IMAGE_TAG -f ./Dockerfile ."
    if [[ "$HTTP_PROXY_HOST" == "your_http_proxy_host" ]] || [[ "$HTTP_PROXY_PORT" == "your_http_proxy_port" ]] || [[ "$HTTPS_PROXY_HOST" == "your_https_proxy_host" ]] || [[ "$HTTPS_PROXY_PORT" == "your_https_proxy_port" ]]
    then
        echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delet the image just created and modify the proxy in the script, then rerun this script."
        $No_Proxy_Modified
    else
        $Proxy_Modified
    fi
fi

