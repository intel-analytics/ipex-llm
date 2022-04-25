#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# echo commands to the terminal output
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
echo "###################################### $SPARK_K8S_CMD"
case "$SPARK_K8S_CMD" in
    driver | driver-py | driver-r | executor)
      shift 1
      ;;
    "")
      ;;
    *)
      echo "Non-spark-on-k8s command provided, proceeding in pass-through mode..."
      exec /usr/bin/tini -s -- "$@"
      ;;
esac

SPARK_CLASSPATH="$SPARK_CLASSPATH:${SPARK_HOME}/jars/*"
env | grep SPARK_JAVA_OPT_ | sort -t_ -k4 -n | sed 's/[^=]*=\(.*\)/\1/g' > /tmp/java_opts.txt
readarray -t SPARK_EXECUTOR_JAVA_OPTS < /tmp/java_opts.txt

if [ -n "$SPARK_EXTRA_CLASSPATH" ]; then
  SPARK_CLASSPATH="$SPARK_CLASSPATH:$SPARK_EXTRA_CLASSPATH"
fi

if [ -n "$PYSPARK_FILES" ]; then
    PYTHONPATH="$PYTHONPATH:$PYSPARK_FILES"
fi

PYSPARK_ARGS=""
if [ -n "$PYSPARK_APP_ARGS" ]; then
    PYSPARK_ARGS="$PYSPARK_APP_ARGS"
fi

R_ARGS=""
if [ -n "$R_APP_ARGS" ]; then
    R_ARGS="$R_APP_ARGS"
fi

if [ "$PYSPARK_MAJOR_PYTHON_VERSION" == "2" ]; then
    pyv="$(python -V 2>&1)"
    export PYTHON_VERSION="${pyv:7}"
    export PYSPARK_PYTHON="python"
    export PYSPARK_DRIVER_PYTHON="python"
elif [ "$PYSPARK_MAJOR_PYTHON_VERSION" == "3" ]; then
    pyv3="$(python3 -V 2>&1)"
    export PYTHON_VERSION="${pyv3:7}"
    export PYSPARK_PYTHON="python3"
    export PYSPARK_DRIVER_PYTHON="python3"
fi

case "$SPARK_K8S_CMD" in
  driver)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@"
    )
    echo $SGX_ENABLED && \
    echo $SGX_MEM_SIZE && \
    echo $SGX_JVM_MEM_SIZE && \
    echo $SGX_LOG_LEVEL && \
    echo $SPARK_EXECUTOR_MEMORY && \
    unset PYTHONHOME && \
    unset PYTHONPATH && \
    if [ "$SGX_ENABLED" == "false" ]; then
        $SPARK_HOME/bin/spark-submit --conf spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS --deploy-mode client "$@"
    elif [ "$SGX_ENABLED" == "true" ]; then
        $SPARK_HOME/bin/spark-submit --conf spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS --deploy-mode client "$@"
    fi
    ;;
  driver-py)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@" $PYSPARK_PRIMARY $PYSPARK_ARGS
    )
    ;;
    driver-r)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@" $R_PRIMARY $R_ARGS
    )
    ;;
    executor)
    echo $SGX_ENABLED && \
    echo $SGX_MEM_SIZE && \
    echo $SGX_JVM_MEM_SIZE && \
    echo $SGX_LOG_LEVEL && \
    echo $SPARK_EXECUTOR_MEMORY && \
    unset PYTHONHOME && \
    unset PYTHONPATH && \
    if [ "$SGX_ENABLED" == "false" ]; then
      /opt/jdk8/bin/java \
        -Xms$SPARK_EXECUTOR_MEMORY \
        -Xmx$SPARK_EXECUTOR_MEMORY \
        "${SPARK_EXECUTOR_JAVA_OPTS[@]}" \
        -cp "$SPARK_CLASSPATH" \
        org.apache.spark.executor.CoarseGrainedExecutorBackend \
        --driver-url $SPARK_DRIVER_URL \
        --executor-id $SPARK_EXECUTOR_ID \
        --cores $SPARK_EXECUTOR_CORES \
        --app-id $SPARK_APPLICATION_ID \
        --hostname $SPARK_EXECUTOR_POD_IP \
        --resourceProfileId $SPARK_RESOURCE_PROFILE_ID
    elif [ "$SGX_ENABLED" == "true" ]; then
      ./init.sh && \
      export spark_commnd="/opt/jdk8/bin/java -Xms$SGX_JVM_MEM_SIZE -Xmx$SGX_JVM_MEM_SIZE "${SPARK_EXECUTOR_JAVA_OPTS[@]}" -cp "$SPARK_CLASSPATH" org.apache.spark.executor.CoarseGrainedExecutorBackend --driver-url $SPARK_DRIVER_URL --executor-id $SPARK_EXECUTOR_ID --cores $SPARK_EXECUTOR_CORES --app-id $SPARK_APPLICATION_ID --hostname $SPARK_EXECUTOR_POD_IP --resourceProfileId $SPARK_RESOURCE_PROFILE_ID" && \
      echo $spark_commnd && \
      SGX=1 ./pal_loader bash -c "export TF_MKL_ALLOC_MAX_BYTES=10737418240 && \
          export _SPARK_AUTH_SECRET=$_SPARK_AUTH_SECRET && \
          $spark_commnd" 1>&2
    fi
    ;;

  *)
    echo "Unknown command: $SPARK_K8S_CMD" 1>&2
    exit 1
esac

# Execute the container CMD under tini for better hygiene
#exec /usr/bin/tini -s -- "${CMD[@]}"
