export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port

Proxy_Modified="sudo docker build \
    --build-arg http_proxy=http://${HTTP_PROXY_HOST}:${HTTP_PROXY_PORT} \
    --build-arg https_proxy=http://${HTTPS_PROXY_HOST}:${HTTPS_PROXY_PORT} \
    --build-arg no_proxy=x.x.x.x \
    --build-arg JDK_URL=your_jdk_url \
    -t intelanalytics/bigdl-ppml-gramine-base:2.4.0 -f ./Dockerfile ."

No_Proxy_Modified="sudo docker build \
    --build-arg no_proxy=x.x.x.x \
    --build-arg JDK_URL=your_jdk_url \
    -t intelanalytics/bigdl-ppml-gramine-base:2.4.0 -f ./Dockerfile ."

if [[ "$HTTP_PROXY_HOST" == "your_http_proxy_host" ]] || [[ "$HTTP_PROXY_PORT" == "your_http_proxy_port" ]] || [[ "$HTTPS_PROXY_HOST" == "your_https_proxy_host" ]] || [[ "$HTTPS_PROXY_PORT" == "your_https_proxy_port" ]]
then
    echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delete the image just created and modify the proxy in the script, then rerun this script."
    $No_Proxy_Modified
    echo "If your environment don't need to set proxy, please ignore this notice information; if your environment need to set proxy, please delete the image just created and modify the proxy in the script, then rerun this script."
else
    $Proxy_Modified
fi
