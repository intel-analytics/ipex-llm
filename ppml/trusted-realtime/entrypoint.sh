#!/usr/bin/env bash
###############################################################################
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

COMMAND_STANDALONE="standalone-job"
COMMAND_HISTORY_SERVER="history-server"

# If unspecified, the hostname of the container is taken as the JobManager address
JOB_MANAGER_RPC_ADDRESS=${JOB_MANAGER_RPC_ADDRESS:-$(hostname -f)}
CONF_FILE="${FLINK_HOME}/conf/flink-conf.yaml"

drop_privs_cmd() {
    if [ $(id -u) != 0 ]; then
        # Don't need to drop privs if EUID != 0
        return
    elif [ -x /sbin/su-exec ]; then
        # Alpine
        echo su-exec flink
    else
        # Others
        echo gosu flink
    fi
}

copy_plugins_if_required() {
  if [ -z "$ENABLE_BUILT_IN_PLUGINS" ]; then
    return 0
  fi

  echo "Enabling required built-in plugins"
  for target_plugin in $(echo "$ENABLE_BUILT_IN_PLUGINS" | tr ';' ' '); do
    echo "Linking ${target_plugin} to plugin directory"
    plugin_name=${target_plugin%.jar}

    mkdir -p "${FLINK_HOME}/plugins/${plugin_name}"
    if [ ! -e "${FLINK_HOME}/opt/${target_plugin}" ]; then
      echo "Plugin ${target_plugin} does not exist. Exiting."
      exit 1
    else
      ln -fs "${FLINK_HOME}/opt/${target_plugin}" "${FLINK_HOME}/plugins/${plugin_name}"
      echo "Successfully enabled ${target_plugin}"
    fi
  done
}

set_config_option() {
  local option=$1
  local value=$2

  # escape periods for usage in regular expressions
  local escaped_option=$(echo ${option} | sed -e "s/\./\\\./g")

  # either override an existing entry, or append a new one
  if grep -E "^${escaped_option}:.*" "${CONF_FILE}" > /dev/null; then
        sed -i -e "s/${escaped_option}:.*/$option: $value/g" "${CONF_FILE}"
  else
        echo "${option}: ${value}" >> "${CONF_FILE}"
  fi
}

prepare_configuration() {
    set_config_option jobmanager.rpc.address ${JOB_MANAGER_RPC_ADDRESS}
    set_config_option blob.server.port 6124
    set_config_option query.server.port 6125

    if [ -n "${TASK_MANAGER_NUMBER_OF_TASK_SLOTS}" ]; then
        set_config_option taskmanager.numberOfTaskSlots ${TASK_MANAGER_NUMBER_OF_TASK_SLOTS}
    fi
    if [ -n "${FLINK_PROPERTIES}" ]; then
        echo "${FLINK_PROPERTIES}" >> "${CONF_FILE}"
    fi
    envsubst < "${CONF_FILE}" > "${CONF_FILE}.tmp" && mv "${CONF_FILE}.tmp" "${CONF_FILE}"
}

maybe_enable_jemalloc() {
    if [ "${DISABLE_JEMALLOC:-false}" == "false" ]; then
        JEMALLOC_PATH="/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so"
        JEMALLOC_FALLBACK="/usr/lib/x86_64-linux-gnu/libjemalloc.so"
        if [ -f "$JEMALLOC_PATH" ]; then
            export LD_PRELOAD=$LD_PRELOAD:$JEMALLOC_PATH
        elif [ -f "$JEMALLOC_FALLBACK" ]; then
            export LD_PRELOAD=$LD_PRELOAD:$JEMALLOC_FALLBACK
        else
            if [ "$JEMALLOC_PATH" = "$JEMALLOC_FALLBACK" ]; then
                MSG_PATH=$JEMALLOC_PATH
            else
                MSG_PATH="$JEMALLOC_PATH and $JEMALLOC_FALLBACK"
            fi
            echo "WARNING: attempted to load jemalloc from $MSG_PATH but the library couldn't be found. glibc will be used instead."
        fi
    fi
}

maybe_enable_jemalloc

copy_plugins_if_required

prepare_configuration

args=("$@")
if [ "$1" = "help" ]; then
    printf "Usage: $(basename "$0") (jobmanager|${COMMAND_STANDALONE}|taskmanager|${COMMAND_HISTORY_SERVER})\n"
    printf "    Or $(basename "$0") help\n\n"
    printf "By default, Flink image adopts jemalloc as default memory allocator. This behavior can be disabled by setting the 'DISABLE_JEMALLOC' environment variable to 'true'.\n"
    exit 0
elif [ "$1" = "jobmanager" ]; then
    args=("${args[@]:1}")

    echo "Starting Job Manager"

    exec $(drop_privs_cmd) "$FLINK_HOME/bin/jobmanager.sh" start-foreground "${args[@]}"
elif [ "$1" = ${COMMAND_STANDALONE} ]; then
    args=("${args[@]:1}")

    echo "Starting Job Manager"

    exec $(drop_privs_cmd) "$FLINK_HOME/bin/standalone-job.sh" start-foreground "${args[@]}"
elif [ "$1" = ${COMMAND_HISTORY_SERVER} ]; then
    args=("${args[@]:1}")

    echo "Starting History Server"

    exec $(drop_privs_cmd) "$FLINK_HOME/bin/historyserver.sh" start-foreground "${args[@]}"
elif [ "$1" = "taskmanager" ]; then
    args=("${args[@]:1}")

    echo "Starting Task Manager"

    exec $(drop_privs_cmd) "$FLINK_HOME/bin/taskmanager.sh" start-foreground "${args[@]}"
elif [ -z "$2" ]; then
    exec "$1"
else
    params=($3)
    if [ "${params[0]}" = "kubernetes-jobmanager.sh" ]; then
        export _FLINK_HOME_DETERMINED='true'
        ARGS=("${params[@]:2}")

        bin=`cd "${FLINK_HOME}/bin"; pwd`
        . "$bin"/config.sh

        if [ "$FLINK_IDENT_STRING" = "" ]; then
            FLINK_IDENT_STRING="$USER"
        fi
        pid=$FLINK_PID_DIR/flink-$FLINK_IDENT_STRING-${params[1]}.pid
        mkdir -p "$FLINK_PID_DIR"
        # The lock needs to be released after use because this script is started foreground
        command -v flock >/dev/null 2>&1
        flock_exist=$?
        if [[ ${flock_exist} -eq 0 ]]; then
            exec 200<"$FLINK_PID_DIR"
            flock 200
        fi

        # Remove the pid file when all the processes are dead
        if [ -f "$pid" ]; then
            all_dead=0
            while read each_pid; do
                # Check whether the process is still running
                kill -0 $each_pid > /dev/null 2>&1
                [[ $? -eq 0 ]] && all_dead=1
            done < "$pid"
            [ ${all_dead} -eq 0 ] && rm $pid
        fi
        id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")

        # Add the current process id to pid file
        echo $$ >> "$pid" 2>/dev/null

        # Release the lock because the java process runs in the foreground and would block other processes from modifying the pid file
        [[ ${flock_exist} -eq 0 ]] &&  flock -u 200

        export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_JM}"
        parseJmArgsAndExportLogs "${ARGS[@]}"

        if [ ! -z "${DYNAMIC_PARAMETERS}" ]; then
            ARGS=(${DYNAMIC_PARAMETERS[@]} "${ARGS[@]}")
        fi

        FLINK_ENV_JAVA_OPTS=$(eval echo ${FLINK_ENV_JAVA_OPTS})

        if [ "${params[1]}" = "kubernetes-application" ]; then
            CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesApplicationClusterEntrypoint
        elif [ "${params[1]}" = "kubernetes-session" ]; then
            CLASS_TO_RUN=org.apache.flink.kubernetes.entrypoint.KubernetesSessionClusterEntrypoint
        fi

        log="${FLINK_HOME}/log/${JOB_MANAGER_RPC_ADDRESS}.log"
        log_setting=("-Dlog.file=${log}" "-Dlog4j.configuration=file:${FLINK_CONF_DIR}/log4j-console.properties" "-Dlog4j.configurationFile=file:${FLINK_CONF_DIR}/log4j-console.properties" "-Dlogback.configurationFile=file:${FLINK_CONF_DIR}/logback-console.xml")

        classpath=$(echo ${FLINK_HOME}/lib/* | tr ' ' ':')

        runtime_command=($JAVA_RUN $JVM_ARGS ${FLINK_ENV_JAVA_OPTS} "${log_setting[@]}" -classpath ${classpath} ${CLASS_TO_RUN} "${ARGS[@]}")
        export sgx_command="${runtime_command[@]}"
        ./init.sh && \
        gramine-sgx bash 1>&2

    elif [ "${params[0]}" = "kubernetes-taskmanager.sh" ]; then

        ARGS=$(eval echo "${params[@]:1}")
        ARGS=(${ARGS[@]})

        export _FLINK_HOME_DETERMINED='true'

        bin=`cd "${FLINK_HOME}/bin"; pwd`
        . "$bin"/config.sh

        if [ "$FLINK_IDENT_STRING" = "" ]; then
            FLINK_IDENT_STRING="$USER"
        fi

        pid=$FLINK_PID_DIR/flink-$FLINK_IDENT_STRING-kubernetes-taskmanager.pid
        mkdir -p "$FLINK_PID_DIR"
        # The lock needs to be released after use because this script is started foreground
        command -v flock >/dev/null 2>&1
        flock_exist=$?
        if [[ ${flock_exist} -eq 0 ]]; then
            exec 200<"$FLINK_PID_DIR"
            flock 200
        fi

        # Remove the pid file when all the processes are dead
        if [ -f "$pid" ]; then
            all_dead=0
            while read each_pid; do
                # Check whether the process is still running
                kill -0 $each_pid > /dev/null 2>&1
                [[ $? -eq 0 ]] && all_dead=1
            done < "$pid"
            [ ${all_dead} -eq 0 ] && rm $pid
        fi
        id=$([ -f "$pid" ] && echo $(wc -l < "$pid") || echo "0")

        # Add the current process id to pid file
        echo $$ >> "$pid" 2>/dev/null

        # Release the lock because the java process runs in the foreground and would block other processes from modifying the pid file
        [[ ${flock_exist} -eq 0 ]] &&  flock -u 200

        if [ -z "${FLINK_ENV_JAVA_OPTS}" ] && [ -z "${FLINK_ENV_JAVA_OPTS_TM}" ]; then
            export JVM_ARGS="$JVM_ARGS -XX:+UseG1GC"
        fi

        export FLINK_ENV_JAVA_OPTS="${FLINK_ENV_JAVA_OPTS} ${FLINK_ENV_JAVA_OPTS_TM}"
        export JVM_ARGS="$JVM_ARGS $FLINK_TM_JVM_MEM_OPTS"

        ARGS=("--configDir" "${FLINK_CONF_DIR}" "${ARGS[@]}")
        FLINK_ENV_JAVA_OPTS=$(eval echo ${FLINK_ENV_JAVA_OPTS})

        CLASS_TO_RUN=org.apache.flink.kubernetes.taskmanager.KubernetesTaskExecutorRunner

        log="${FLINK_HOME}/log/${TASK_MANAGER_RPC_ADDRESS}.log"
        log_setting=("-Dlog.file=${log}" "-Dlog4j.configuration=file:${FLINK_CONF_DIR}/log4j-console.properties" "-Dlog4j.configurationFile=file:${FLINK_CONF_DIR}/log4j-console.properties" "-Dlogback.configurationFile=file:${FLINK_CONF_DIR}/logback-console.xml")

        classpath=$(echo ${FLINK_HOME}/lib/* | tr ' ' ':')

        runtime_command=($JAVA_RUN $JVM_ARGS "-XX:CompressedClassSpaceSize=64m" ${FLINK_ENV_JAVA_OPTS} "${log_setting[@]}" -classpath ${classpath} ${CLASS_TO_RUN} "${ARGS[@]}")
        export sgx_command="${runtime_command[@]}"
        ./init.sh && \
        gramine-sgx bash 1>&2
    fi
fi
