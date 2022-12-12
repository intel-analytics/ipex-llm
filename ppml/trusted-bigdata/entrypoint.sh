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

# Set PCCS conf
if [ "$PCCS_URL" != "" ]; then
  echo 'PCCS_URL='${PCCS_URL}'/sgx/certification/v3/' >/etc/sgx_default_qcnl.conf
  echo 'USE_SECURE_CERT=FALSE' >>/etc/sgx_default_qcnl.conf
fi

# If there is no passwd entry for the container UID, attempt to create one
if [ -z "$uidentry" ]; then
  if [ -w /etc/passwd ]; then
    echo "$myuid:x:$myuid:$mygid:anonymous uid:$SPARK_HOME:/bin/false" >>/etc/passwd
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
"") ;;

*)
  echo "Non-spark-on-k8s command provided, proceeding in pass-through mode..."
  exec /usr/bin/tini -s -- "$@"
  ;;
esac

SPARK_CLASSPATH="$SPARK_CLASSPATH:${SPARK_HOME}/jars/*"
env | grep SPARK_JAVA_OPT_ | sort -t_ -k4 -n | sed 's/[^=]*=\(.*\)/\1/g' >/tmp/java_opts.txt
readarray -t SPARK_EXECUTOR_JAVA_OPTS </tmp/java_opts.txt

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

# Attestation
if [ -z "$ATTESTATION" ]; then
  echo "[INFO] Attestation is disabled!"
  ATTESTATION="false"
elif [ "$ATTESTATION" = "true" ]; then
  echo "[INFO] Attestation is enabled!"
  # Build ATTESTATION_COMMAND
  if [ -z "$ATTESTATION_URL" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_URL is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$ATTESTATION_ID" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_ID is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  if [ -z "$ATTESTATION_KEY" ]; then
    echo "[ERROR] Attestation is enabled, but ATTESTATION_KEY is empty!"
    echo "[INFO] PPML Application Exit!"
    exit 1
  fi
  ATTESTATION_COMMAND="/opt/jdk8/bin/java -Xmx1g -cp $SPARK_CLASSPATH:$BIGDL_HOME/jars/* com.intel.analytics.bigdl.ppml.attestation.AttestationCLI -u ${ATTESTATION_URL} -i ${ATTESTATION_ID}  -k ${ATTESTATION_KEY}"
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
  echo $SGX_ENABLED &&
    echo $SGX_DRIVER_MEM_SIZE &&
    echo $SGX_DRIVER_JVM_MEM_SIZE &&
    echo $SGX_EXECUTOR_MEM_SIZE &&
    echo $SGX_EXECUTOR_JVM_MEM_SIZE &&
    echo $SGX_LOG_LEVEL &&
    echo $SPARK_DRIVER_MEMORY &&
    unset PYTHONHOME &&
    unset PYTHONPATH &&
    if [ "$SGX_ENABLED" == "false" ]; then
      $SPARK_HOME/bin/spark-submit --conf spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS --deploy-mode client "$@"
    elif [ "$SGX_ENABLED" == "true" ]; then
      export driverExtraClassPath=$(cat /opt/spark/conf/spark.properties | grep -P -o "(?<=spark.driver.extraClassPath=).*") &&
        echo $driverExtraClassPath &&
        export SGX_MEM_SIZE=$SGX_DRIVER_MEM_SIZE &&
        export sgx_command="/opt/jdk8/bin/java -Dlog4j.configurationFile=/ppml/spark-${SPARK_VERSION}/conf/log4j2.xml -Xms1G -Xmx$SGX_DRIVER_JVM_MEM_SIZE -cp "$SPARK_CLASSPATH:$driverExtraClassPath" org.apache.spark.deploy.SparkSubmit --conf spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS --deploy-mode client "$@"" &&
        if [ "$ATTESTATION" = "true" ]; then
          # Also consider ENCRYPTEDFSD condition
          rm /ppml/temp_command_file || true
          bash attestation.sh
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
          fi
          echo $sgx_command >>temp_command_file
          export sgx_command="bash temp_command_file && rm temp_command_file"
        else
          # ATTESTATION is false
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            # ATTESTATION false, encrypted-fsd true
            rm /ppml/temp_command_file || true
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
            echo $sgx_command >>temp_command_file
            export sgx_command="bash temp_command_file && rm temp_command_file"
          fi
        fi
      echo $sgx_command &&
        ./init.sh &&
        gramine-sgx bash 1>&2
      rm /ppml/temp_command_file || true
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
  echo $SGX_ENABLED &&
    echo $SGX_DRIVER_MEM_SIZE &&
    echo $SGX_DRIVER_JVM_MEM_SIZE &&
    echo $SGX_EXECUTOR_MEM_SIZE &&
    echo $SGX_EXECUTOR_JVM_MEM_SIZE &&
    echo $SGX_LOG_LEVEL &&
    echo $SPARK_EXECUTOR_MEMORY &&
    unset PYTHONHOME &&
    unset PYTHONPATH &&
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
      export SGX_MEM_SIZE=$SGX_EXECUTOR_MEM_SIZE
      export sgx_command="/opt/jdk8/bin/java -Dlog4j.configurationFile=/ppml/spark-${SPARK_VERSION}/conf/log4j2.xml -Xms1G -Xmx$SGX_EXECUTOR_JVM_MEM_SIZE "${SPARK_EXECUTOR_JAVA_OPTS[@]}" -cp "$SPARK_CLASSPATH" org.apache.spark.executor.CoarseGrainedExecutorBackend --driver-url $SPARK_DRIVER_URL --executor-id $SPARK_EXECUTOR_ID --cores $SPARK_EXECUTOR_CORES --app-id $SPARK_APPLICATION_ID --hostname $SPARK_EXECUTOR_POD_IP --resourceProfileId $SPARK_RESOURCE_PROFILE_ID" &&
        if [ "$ATTESTATION" = "true" ]; then
          # Also consider ENCRYPTEDFSD condition
          rm /ppml/temp_command_file || true
          bash attestation.sh
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
          fi
          echo $sgx_command >>temp_command_file
          export sgx_command="bash temp_command_file && rm temp_command_file"
        else
          # ATTESTATION is false
          if [ "$ENCRYPTED_FSD" == "true" ]; then
            # ATTESTATION false, encrypted-fsd true
            rm /ppml/temp_command_file || true
            echo "[INFO] Distributed encrypted file system is enabled"
            bash encrypted-fsd.sh
            echo $sgx_command >>temp_command_file
            export sgx_command="bash temp_command_file && rm temp_command_file"
          fi
        fi
      echo $sgx_command &&
        ./init.sh &&
        gramine-sgx bash 1>&2
      rm /ppml/temp_command_file || true
    fi
  ;;

*)
  echo "Unknown command: $SPARK_K8S_CMD" 1>&2
  exit 1
  ;;
esac

# Execute the container CMD under tini for better hygiene
#exec /usr/bin/tini -s -- "${CMD[@]}"
