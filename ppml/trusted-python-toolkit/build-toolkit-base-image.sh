export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export BASE_IMAGE_NAME=your_base_image_name
export BASE_IMAGE_TAG=your_base_image_tag
export TOOLKIT_IMAGE_NAME=your_toolkit_image_name
export TOOLKIT_IMAGE_TAG=your_toolkit_image_tag

Proxy_Modified="sudo docker build \
    --build-arg http_proxy=${HTTP_PROXY_HOST}:${HTTP_PROXY_PORT} \
    --build-arg https_proxy=${HTTPS_PROXY_HOST}:${HTTP_PROXY_PORT} \
    --build-arg BASE_IMAGE_NAME=$BASE_IMAGE_NAME \
    --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
    -t ${TOOLKIT_IMAGE_NAME}:${TOOLKIT_IMAGE_TAG} -f ./Dockerfile ."

No_Proxy_Modified="sudo docker build \
    --build-arg BASE_IMAGE_NAME=$BASE_IMAGE_NAME \
    --build-arg BASE_IMAGE_TAG=$BASE_IMAGE_TAG \
    -t ${FL_IMAGE_NAME}:${FL_IMAGE_TAG} -f ./Dockerfile ."

if [[ "$HTTP_PROXY_HOST" == "your_http_proxy_host" ]] || [[ "$HTTP_PROXY_PORT" == "your_http_proxy_port" ]] || [[ "$HTTPS_PROXY_HOST" == "your_https_proxy_host" ]] || [[ "$HTTPS_PROXY_PORT" == "your_https_proxy_port" ]]
then
   echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delet the image just created and modify the proxy in the script, then rerun this script."
   $No_Proxy_Modified
else
   $Proxy_Modified
fi

