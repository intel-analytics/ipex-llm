export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export JDK_URL=http://your-http-url-to-download-jdk
export IMAGE_MODE=user_image
export BIGDL_IMAGE_NAME=10.239.45.10/arda/intelanalytics/mrenclave-bigdl-image
export BIGDL_IMAGE_VERSION=latest
export USER_IMAGE_TAG=user_image_1.0
export LOCAL_IP=10.239.44.70

if [ "$IMAGE_MODE" == "bigdl_base_image_or_user_image" ]
then
    echo "Please modify the mode of image to build, you can choose bigdl_base_image or user_image. The user_image is built on top of bigdl_base_image, where the user_image is a user-specific one signed by your enclave-key.pem."
else
    if [ "$IMAGE_MODE" == "bigdl_base_image" ]
    then
        Proxy_Modified="sudo docker build \
            --build-arg http_proxy=http://${HTTP_PROXY_HOST}:${HTTP_PROXY_PORT} \
            --build-arg https_proxy=http://${HTTPS_PROXY_HOST}:${HTTPS_PROXY_PORT} \
            --build-arg HTTP_PROXY_HOST=${HTTP_PROXY_HOST} \
            --build-arg HTTP_PROXY_PORT=${HTTP_PROXY_PORT} \
            --build-arg HTTPS_PROXY_HOST=${HTTPS_PROXY_HOST} \
            --build-arg HTTPS_PROXY_PORT=${HTTPS_PROXY_PORT} \
            --build-arg JDK_VERSION=8u192 \
            --build-arg JDK_URL=${JDK_URL} \
            --build-arg no_proxy=x.x.x.x \
            -t intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT -f ./Dockerfile ."

        No_Proxy_Modified="sudo docker build \
            --build-arg JDK_VERSION=8u192 \
            --build-arg JDK_URL=${JDK_URL} \
            --build-arg no_proxy=x.x.x.x \
            -t intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT -f ./Dockerfile ."

        if [ "$JDK_URL" == "http://your-http-url-to-download-jdk" ]
        then
            echo "Please modify the path of JDK_URL to the suitable url in this script, then rerun this script. And if your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please modify the proxy in the script, then rerun this script."
        else
            if [[ "$HTTP_PROXY_HOST" == "your_http_proxy_host" ]] || [[ "$HTTP_PROXY_PORT" == "your_http_proxy_port" ]] || [[ "$HTTPS_PROXY_HOST" == "your_https_proxy_host" ]] || [[ "$HTTPS_PROXY_PORT" == "your_https_proxy_port" ]]
            then
                echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delet the image just created and modify the proxy in the script, then rerun this script."
                $No_Proxy_Modified
                echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delet the image just created and modify the proxy in the script, then rerun this script."
            else
                $Proxy_Modified
            fi
        fi
    else
    if [ "$IMAGE_MODE" == "user_image" ]
    then
        if [ "$BIGDL_IMAGE_NAME" == "your_bigdl_base_image_name_used_to_build_user_image" ] || [ "$BIGDL_IMAGE_VERSION" == "your_bigdl_base_image_version_used_to_build_user_image" ] || [ "$USER_IMAGE_TAG" == "your_user_image_tag" ]
        then
            echo "Please specific name and version of your bigdl base name, and tag of your user image to use."
        else
            sudo docker build \
                    --build-arg BIGDL_IMAGE_NAME=$BIGDL_IMAGE_NAME \
                    --build-arg BIGDL_IMAGE_VERSION=$BIGDL_IMAGE_VERSION \
                    --build-arg LOCAL_IP=$LOCAL_IP \
                    -t $USER_IMAGE_TAG -f ./UserImageDockerfile .
        fi
     fi
  fi
fi

