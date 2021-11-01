#!/bin/bash
# Acceptable arguments: redis, flinkjm, flinktm, frontend, serving, all

REDISLOG="/ppml/trusted-realtime-ml/redis/redis-${SGX_MODE}.log"
JMSGXLOG="/ppml/trusted-realtime-ml/java/flink-jobmanager-${SGX_MODE}.log"
STANDALONELOG="/ppml/trusted-realtime-ml/java/work/flink-${FLINK_VERSION}/log/flink-standalonesession-*.log"
TMSGXLOG="/ppml/trusted-realtime-ml/java/work/flink-${FLINK_VERSION}/log/flink-taskexecutor-*.log"
FRONTENDLOG="/ppml/trusted-realtime-ml/java/http-frontend-${SGX_MODE}.log"
SERVINGLOG="/ppml/trusted-realtime-ml/java/cluster-serving-job-${SGX_MODE}.log"

redis () {
    echo "Detecting redis status..."
    REDISSUCCESS=""
    test -f  $REDISLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find redis log at" $REDISLOG 
    else 
        REDISSUCCESS=$(cat $REDISLOG | grep "Ready to accept connections")
        if [ -z "$REDISSUCCESS" ] ; then
            echo "Redis initialization failed. See" $REDISLOG " for details."
            echo "To restart Redis, run /ppml/trusted-realtime-ml/redis/start-redis.sh in the docker container."
        fi
    fi
    REDISPORT=$(netstat -nlp | grep 6379)
    # Default redis port is 6379
    if [ -z "$REDISPORT" ]; then
        echo "Redis initialization failed. Unable to get redis port at " $REDIS_PORT "."
    fi
    
    if [ -n "$REDISPORT" ] && [ -n "$REDISSUCCESS" ] ; then
        echo "Redis initialization successful."
    fi
}

flinkjm () {
    echo "Detecting Flink job manager status..."
    JMSUCCESS=""
    test -f $JMSGXLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find flink-jobmanager-sgx.log at path" $JMSGXLOG 
    fi
    test -f $STANDALONELOG 
    if [ $? -eq 1 ]; then 
        echo "Cannot find standalonesession log at path" $STANDALONELOG
    else 
        JMSUCCESS=$(cat $STANDALONELOG | grep "Successfully recovered 0 persisted job graphs.")
        if [ -z "$JMSUCCESS" ] ; then
            echo "Flink job manager initialization failed. See" $STANDALONELOG "for details."
            echo "To restart Flink job manager, run /ppml/trusted-realtime-ml/java/start-flink-jobmanager.sh. in the docker container."
        fi
    fi
    JMPORT=$(netstat -nlp | grep 8081)
    # Default jm port is 8081.
    if [ -z "$JMPORT" ]; then
        echo "Flink job manager initialization failed. Unable to get Flink job manager rest port at " $FLINK_JOB_MANAGER_REST_PORT "."
    fi
    
    if [ -n "$JMPORT" ] && [ -n "$JMSUCCESS" ] ; then
        echo "Flink job manager initialization successful."
    fi
}

flinktm () {
    echo "Detecting Flink task manager status..."
    TMSUCCESS=""
    test -f $TMSGXLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find Flink task manager log at path" $TMSGXLOG
    else 
        TMSUCCESS=$(cat $TMSGXLOG | grep "Successful registration at reousrce manager")
        if [ -z "$TMSUCCESS" ] ; then
            echo "Flink task manager initialization failed. See" $TMSGXLOG "for details."
            echo "To restart Flink task manager, run /ppml/trusted-realtime-ml/java/start-flink-taskmanager.sh in the docker container."
        fi
    fi
    TMPORT=$(netstat -nlp | grep 6123)
    # Default TM port is 6123.
    if [ -z "$FLINK_TASK_MANAGER_DATA_PORT" ]; then
        echo "Flink task manager initialization failed. Unable to get Flink task manager data port at " $FLINK_TASK_MANAGER_DATA_PORT "."
    fi
    
    if [ -n "$TMPORT" ] && [ -n "$TMSUCCESS" ] ; then
        echo "Flink task manager initialization successful."
    fi
}

frontend () {
    echo "Detecting http frontend status. This may take a while."
    test -f "$FRONTENDLOG"
    if [ $? -eq 1 ] ; then
        echo "Cannot find http frontend log at path" $FRONTENDLOG 
    else 
        FRONTENDSUCCESS=$(cat $FRONTENDLOG | grep "https started at https://0.0.0.0:10023")
        if [ -z "$FRONTENDSUCCESS" ] ; then
            echo "Http frontend initialization failed. See" $FRONTENDLOG "for details."
            echo "To restart http frontend, run /ppml/trusted-realtime-ml/java/start-http-frontend.sh in the docker container."
        else 
            echo "Http frontend initialization successful."
        fi
    fi
}

serving () {
    echo "Detecting cluster-serving-job status..."
    test -f "$SERVINGLOG" 
    if [ $? -eq 1 ] ; then
        echo "Cannot find cluster-serving-job log at path" $SERVINGLOG 
    else 
        SERVINGSUCCESS=$(cat $SERVINGLOG | grep "Job has been submitted with JobID")
        if [ -z "$SERVINGSUCCESS" ] ; then
            echo "cluster-serving-job initialization failed. See" $SERVINGLOG "for details."
            echo "To restart cluster-serving-job, run /ppml/trusted-realtime-ml/java/start-cluster-serving-job.sh in the docker container."
        else 
            echo "cluster-serving-job initialization successful."
        fi
    fi
}


all=0
if [ "$#" -lt 1 ]; then
    echo "No argument passed, detecting all component statuses."
    all=$((all+1))
else
    for arg in "$@"
    do
        if [ "$arg" == all ]; then
            echo "Detecting all component statuses."
            all=$((all+1))
        fi
    done
fi


if [ "$#" -gt 5 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"redis\", \"flinkjm\", \"flinktm\", \"frontend\", \"serving\""
elif [ "$all" -eq 1 ]; then 
    redis
    flinkjm
    flinktm
    frontend
    serving
else 
    for arg in "$@"
    do
        if [ "$arg" == redis ]; then
            redis
        elif [ "$arg" == flinkjm ]; then
            flinkjm
        elif [ "$arg" == flinktm ]; then
            flinktm
        elif [ "$arg" == frontend ]; then
            frontend
        elif [ "$arg" == serving ]; then
            serving
        else 
            echo "Acceptable arguments: \"all\", or one or more among \"redis\", \"flinkjm\", \"flinktm\", \"frontend\", \"serving\""
        fi
    done
fi
