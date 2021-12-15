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

SPARK_CLASSPATH="$SPARK_CLASSPATH:/bin/jars/*:/opt/spark/jars/*"
env | grep SPARK_JAVA_OPT_ | sort -t_ -k4 -n | sed 's/[^=]*=\(.*\)/\1/g' > /tmp/java_opts.txt
readarray -t SPARK_EXECUTOR_JAVA_OPTS < /tmp/java_opts.txt

if [ -n "$SPARK_EXTRA_CLASSPATH" ]; then
  SPARK_CLASSPATH="$SPARK_CLASSPATH:$SPARK_EXTRA_CLASSPATH"
fi


/opt/occlum/start_aesm.sh
case "$SPARK_K8S_CMD" in
  driver)
    echo "SGX Mem $SGX_MEM_SIZE"
    /opt/run_spark_on_occlum_glibc.sh init
    cd /opt/occlum_spark
    DMLC_TRACKER_URI=$SPARK_DRIVER_BIND_ADDRESS
    CMD=(
        /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
        -Divy.home="/tmp/.ivy" \
        -Dos.name="Linux" \
        -XX:-UseCompressedOops \
        -XX:MaxMetaspaceSize=256m \
        -Djdk.lang.Process.launchMechanism=posix_spawn \
        -cp "$SPARK_CLASSPATH" \
        -Xms8g \
        -Xmx8g \
        -XX:ActiveProcessorCount=4 \
        -Dio.netty.availableProcessors=64 \
        org.apache.spark.deploy.SparkSubmit \
        --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS" \
        --deploy-mode client \
        "$@"
        )
    ;;
  executor)
    echo "SGX Mem $SGX_MEM_SIZE"
    /opt/run_spark_on_occlum_glibc.sh init
    cd /opt/occlum_spark
    DMLC_TRACKER_URI=$SPARK_DRIVER_BIND_ADDRESS
    CMD=(
        /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
        "${SPARK_EXECUTOR_JAVA_OPTS[@]}" \
        -XX:-UseCompressedOops \
        -XX:MaxMetaspaceSize=256m \
        -XX:ActiveProcessorCount=$SPARK_EXECUTOR_CORES \
        -Divy.home=/tmp/.ivy \
        -Xms$SPARK_EXECUTOR_MEMORY \
        -Xmx$SPARK_EXECUTOR_MEMORY \
        -Dos.name=Linux \
        -Dio.netty.availableProcessors=64 \
        -Djdk.lang.Process.launchMechanism=posix_spawn \
        -cp "$SPARK_CLASSPATH" \
        org.apache.spark.executor.CoarseGrainedExecutorBackend \
        --driver-url $SPARK_DRIVER_URL \
        --executor-id $SPARK_EXECUTOR_ID \
        --cores $SPARK_EXECUTOR_CORES \
        --app-id $SPARK_APPLICATION_ID \
        --hostname $SPARK_EXECUTOR_POD_IP
        )
    ;;

  *)
    echo "Unknown command: $SPARK_K8S_CMD" 1>&2
    exit 1
esac

/sbin/tini -s -- occlum run "${CMD[@]}"
