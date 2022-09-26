export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export JDK_URL=http://your-http-url-to-download-jdk
export SPARK_JAR_REPO_URL=http://your_spark_jar_repo_url
export BIGDL_IMAGE_NAME=your_bigdl_base_image_name_used_to_build_custom_image
export BIGDL_IMAGE_VERSION=your_bigdl_base_image_version_used_to_build_customer_image
export LOCAL_IP=your_local_IP

if [[ "$BIGDL_IMAGE_NAME" == "your_bigdl_base_image_name_used_to_build_custom_image" ]] || [[ "$BIGDL_IMAGE_VERSION" == "your_bigdl_base_image_version_used_to_build_custom_image" ]]
then
    echo "Please specific name and version of your bigdl base name."
else
    Proxy_Modified="sudo docker build \
        --build-arg http_proxy=http://${HTTP_PROXY_HOST}:${HTTP_PROXY_PORT} \
        --build-arg https_proxy=http://${HTTPS_PROXY_HOST}:${HTTPS_PROXY_PORT} \
        --build-arg HTTP_PROXY_HOST=${HTTP_PROXY_HOST} \
        --build-arg HTTP_PROXY_PORT=${HTTP_PROXY_PORT} \
        --build-arg HTTPS_PROXY_HOST=${HTTPS_PROXY_HOST} \
        --build-arg HTTPS_PROXY_PORT=${HTTPS_PROXY_PORT} \
        --build-arg JDK_VERSION=8u192 \
        --build-arg JDK_URL=${JDK_URL} \
        --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
        --build-arg no_proxy=${LOCAL_IP} \
        -t ${BIGDL_IMAGE_NAME}:${BIGDL_IMAGE_VERSION} -f ./Dockerfile ."

    No_Proxy_Modified="sudo docker build \
        --build-arg JDK_VERSION=8u192 \
        --build-arg JDK_URL=${JDK_URL} \
        --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
        --build-arg no_proxy=${LOCAL_IP} \
        -t ${BIGDL_IMAGE_NAME}:${BIGDL_IMAGE_VERSION} -f ./Dockerfile ."

    if [[ "$JDK_URL" == "http://your-http-url-to-download-jdk" ]] || [[ "$SPARK_JAR_REPO_URL" == "http://your_spark_jar_repo_url" ]]
    then
        echo "Please modify the path of JDK_URL and SPARK_JAR_REPO_URL to the suitable url in this script, then rerun this script. And if your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please modify the proxy in the script, then rerun this script."
    else
        if [[ "$HTTP_PROXY_HOST" == "your_http_proxy_host" ]] || [[ "$HTTP_PROXY_PORT" == "your_http_proxy_port" ]] || [[ "$HTTPS_PROXY_HOST" == "your_https_proxy_host" ]] || [[ "$HTTPS_PROXY_PORT" == "your_https_proxy_port" ]]
        then
           echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delet the image just created and modify the proxy in the script, then rerun this script."
           $No_Proxy_Modified
        else
           $Proxy_Modified
        fi
    fi
fi
