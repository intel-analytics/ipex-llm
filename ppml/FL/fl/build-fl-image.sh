export HTTP_PROXY_HOST=http://child-prc.intel.com
export HTTP_PROXY_PORT=913
export HTTPS_PROXY_HOST=http://child-prc.intel.com
export HTTPS_PROXY_PORT=913
export JDK_URL=http://10.239.45.10:8081/repository/raw/jdk/jdk-8u192-linux-x64.tar.gz
export SPARK_JAR_REPO_URL=http://10.239.45.10:8081/repository/raw/spark
export LOCAL_IP=10.239.44.70 
export BASE_IMAGE_NAME=10.239.45.10/arda/intelanalytics/bigdl-ppml-gramine-base
export BASE_IMAGE_TAG=2.2.0-SNAPSHOT
export FL_IMAGE_NAME=intelanalytics/bigdl-ppml-gramine-fl
export FL_IMAGE_TAG=1

Proxy_Modified="sudo docker build \
    --build-arg http_proxy=http://child-prc.intel.com:913/ \
    --build-arg https_proxy=http://child-prc.intel.com:913/ \
    --build-arg HTTP_PROXY_HOST=http://child-prc.intel.com \
    --build-arg HTTP_PROXY_PORT=913 \
    --build-arg HTTPS_PROXY_HOST=http://child-prc.intel.com \
    --build-arg HTTPS_PROXY_PORT=913 \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=http://10.239.45.10:8081/repository/raw/jdk/jdk-8u192-linux-x64.tar.gz \
    --build-arg no_proxy=${LOCAL_IP} \
    --build-arg SPARK_HOME=/opt/spark \
    --build-arg SPARK_VERSION=3.1.2 \
    --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
    -t ${FL_IMAGE_NAME}:${FL_IMAGE_TAG} -f ./Dockerfile ."

No_Proxy_Modified="sudo docker build \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=${JDK_URL} \
    --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
    --build-arg no_proxy=${LOCAL_IP} \
    --build-arg SPARK_HOME=/opt/spark \
    --build-arg SPARK_VERSION=3.1.2 \
    --build-arg SPARK_JAR_REPO_URL=${SPARK_JAR_REPO_URL} \
    -t ${FL_IMAGE_NAME}:${FL_IMAGE_TAG} -f ./Dockerfile ."


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


