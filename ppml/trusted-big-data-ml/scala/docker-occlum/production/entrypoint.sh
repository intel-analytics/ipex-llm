#!/bin/bash
set -ex

# Check whether there is a passwd entry for the container UID
myuid=$(id -u)
mygid=$(id -g)
# turn off -e for getent because it will return error code in anonymous uid case
set +e
uidentry=$(getent passwd $myuid)
set -e

# If there is no passwd entry for the container UID, attempt to create one
if [ -z "$uidentry" ] ; then
    if [ -w /etc/passwd ] ; then
        echo "$myuid:x:$myuid:$mygid:anonymous uid:$SPARK_HOME:/bin/false" >> /etc/passwd
    else
        echo "Container ENTRYPOINT failed to add passwd entry for anonymous UID"
    fi
fi

# check occlum log level for k8s
export ENABLE_SGX_DEBUG=false
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

# check the NETTY_THREAD
if [[ -z "$NETTY_THREAD" ]]; then
    echo "NETTY_THREAD not set, using default value 16"
    NETTY_THREAD=16
fi

SPARK_K8S_CMD="$1"
case "$SPARK_K8S_CMD" in
    driver | executor)
      shift 1
      ;;
    "")
      ;;
    *)
      echo "Non-spark-on-k8s command provided, proceeding in pass-through mode..."
      exec /sbin/tini -s -- "$@"
      ;;
esac

SPARK_CLASSPATH="$SPARK_CLASSPATH:/opt/spark/jars/*:/bin/jars/*"
env | grep SPARK_JAVA_OPT_ | sort -t_ -k4 -n | sed 's/[^=]*=\(.*\)/\1/g' > /tmp/java_opts.txt
readarray -t SPARK_EXECUTOR_JAVA_OPTS < /tmp/java_opts.txt

if [ -n "$SPARK_EXTRA_CLASSPATH" ]; then
  SPARK_CLASSPATH="$SPARK_CLASSPATH:$SPARK_EXTRA_CLASSPATH"
fi

if [[ -z "$META_SPACE" ]]; then
    echo "META_SPACE not set, using default value 256m"
    META_SPACE=256m
else
    echo "META_SPACE=$META_SPACE"
fi

echo "SGX_LOG_LEVEL $SGX_LOG_LEVEL" && \
echo "SGX_DRIVER_JVM_MEM_SIZE $SGX_DRIVER_JVM_MEM_SIZE" && \
echo "SGX_EXECUTOR_JVM_MEM_SIZE $SGX_EXECUTOR_JVM_MEM_SIZE" && \
echo "SPARK_DRIVER_MEMORY $DRIVER_MEMORY" && \
echo "SPARK_EXECUTOR_MEMORY $SPARK_EXECUTOR_MEMORY" && \

case "$SPARK_K8S_CMD" in
  driver)
    echo "SGX Mem $SGX_MEM_SIZE"
    if [[ -z "$DRIVER_MEMORY" ]]; then
        echo "DRIVER_MEMORY not set, using default value 10g"
        DRIVER_MEMORY=10g
    else
        echo "DRIVER_MEMORY=$DRIVER_MEMORY"
    fi
    /opt/run_spark_on_occlum_glibc.sh init
    cd /opt/occlum_spark
    DMLC_TRACKER_URI=$SPARK_DRIVER_BIND_ADDRESS

    if [[ -z "$SGX_DRIVER_JVM_MEM_SIZE" ]]; then
        echo "SGX_DRIVER_JVM_MEM_SIZE not set, using default DRIVER_MEMORY"
        CMD=(
            /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
            -Divy.home="/tmp/.ivy" \
            -Dos.name="Linux" \
            -XX:-UseCompressedOops \
            -XX:MaxMetaspaceSize=$META_SPACE \
            -Djdk.lang.Process.launchMechanism=posix_spawn \
            -cp "$SPARK_CLASSPATH" \
            -Xmx$DRIVER_MEMORY \
            -XX:ActiveProcessorCount=4 \
            -Dio.netty.availableProcessors=$NETTY_THREAD \
            org.apache.spark.deploy.SparkSubmit \
            --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS" \
            --deploy-mode client \
            "$@"
        )
    else
        echo "use SGX_DRIVER_JVM_MEM_SIZE=$SGX_DRIVER_JVM_MEM_SIZE"
        CMD=(
            /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
            -Divy.home="/tmp/.ivy" \
            -Dos.name="Linux" \
            -XX:-UseCompressedOops \
            -XX:MaxMetaspaceSize=$META_SPACE \
            -Djdk.lang.Process.launchMechanism=posix_spawn \
            -cp "$SPARK_CLASSPATH" \
            -Xmx$SGX_DRIVER_JVM_MEM_SIZE \
            -XX:ActiveProcessorCount=4 \
            -Dio.netty.availableProcessors=$NETTY_THREAD \
            org.apache.spark.deploy.SparkSubmit \
            --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS" \
            --deploy-mode client \
            "$@"
        )
    fi
    ;;
  executor)
    echo "SGX Mem $SGX_MEM_SIZE"
    /opt/run_spark_on_occlum_glibc.sh init
    cd /opt/occlum_spark
    DMLC_TRACKER_URI=$SPARK_DRIVER_BIND_ADDRESS

    if [[ -z "$SGX_EXECUTOR_JVM_MEM_SIZE" ]]; then
        echo "SGX_EXECUTOR_JVM_MEM_SIZE not set, using default EXCUTOR_MEMORY"
        CMD=(
            /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
            "${SPARK_EXECUTOR_JAVA_OPTS[@]}" \
            -XX:-UseCompressedOops \
            -XX:MaxMetaspaceSize=$META_SPACE \
            -XX:ActiveProcessorCount=$SPARK_EXECUTOR_CORES \
            -Divy.home=/tmp/.ivy \
            -Xms$SPARK_EXECUTOR_MEMORY \
            -Xmx$SPARK_EXECUTOR_MEMORY \
            -Dos.name=Linux \
            -Dio.netty.availableProcessors=$NETTY_THREAD \
            -Djdk.lang.Process.launchMechanism=posix_spawn \
            -cp "$SPARK_CLASSPATH" \
            org.apache.spark.executor.CoarseGrainedExecutorBackend \
            --driver-url $SPARK_DRIVER_URL \
            --executor-id $SPARK_EXECUTOR_ID \
            --cores $SPARK_EXECUTOR_CORES \
            --app-id $SPARK_APPLICATION_ID \
            --hostname $SPARK_EXECUTOR_POD_IP
            )
    else
        echo "use SGX_EXECUTOR_JVM_MEM_SIZE=$SGX_EXECUTOR_JVM_MEM_SIZE"
        CMD=(
            /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
            "${SPARK_EXECUTOR_JAVA_OPTS[@]}" \
            -XX:-UseCompressedOops \
            -XX:MaxMetaspaceSize=$META_SPACE \
            -XX:ActiveProcessorCount=$SPARK_EXECUTOR_CORES \
            -Divy.home=/tmp/.ivy \
            -Xms$SGX_EXECUTOR_JVM_MEM_SIZE \
            -Xmx$SGX_EXECUTOR_JVM_MEM_SIZE \
            -Dos.name=Linux \
            -Dio.netty.availableProcessors=$NETTY_THREAD \
            -Djdk.lang.Process.launchMechanism=posix_spawn \
            -cp "$SPARK_CLASSPATH" \
            org.apache.spark.executor.CoarseGrainedExecutorBackend \
            --driver-url $SPARK_DRIVER_URL \
            --executor-id $SPARK_EXECUTOR_ID \
            --cores $SPARK_EXECUTOR_CORES \
            --app-id $SPARK_APPLICATION_ID \
            --hostname $SPARK_EXECUTOR_POD_IP
        )
    fi
    ;;
  *)
    echo "Unknown command: $SPARK_K8S_CMD" 1>&2
    exit 1
esac

/sbin/tini -s -- occlum run "${CMD[@]}"
