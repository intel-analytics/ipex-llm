#!/bin/bash
# Acceptable arguments: master, worker, all

MASTERLOG="/ppml/trusted-big-data-ml/spark-master-sgx.log"
WORKERLOG="/ppml/trusted-big-data-ml/spark-worker-sgx.log"

master () {
    echo "(1/2) Detecting master state..."
    MASTERSUCCESS=""
    test -f  $MASTERLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find master log at" $MASTERLOG 
    else 
        MASTERSUCCESS=$(cat $MASTERLOG | grep "I have been elected leader")
        if [ -z "$MASTERSUCCESS" ] ; then
            echo "Master initialization failed. See" $MASTERLOG " for details."
            echo "To restart Master, run ./start-spark-standalone-master-sgx.sh in the docker container."
        fi
    fi
    MASTERPORT=$(netstat -nlp | grep 8080)
    # Default master port is 8080
    if [ -z "$MASTERPORT" ]; then
        echo "Master initialization failed. Unable to get master port at " $MASTERPORT "."
    fi
    
    if [ -n "$MASTERPORT" ] && [ -n "$MASTERSUCCESS" ] ; then
        echo "Master initialization successful."
    fi
}

worker () {
    echo "(2/2) Detecting worker state..."
    WORKERSUCCESS=""
    test -f  $WORKERLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find worker log at" $WORKERLOG 
    else 
        WORKERSUCCESS=$(cat $WORKERLOG | grep "Successfully registered with master")
        if [ -z "$WORKERSUCCESS" ] ; then
            echo "Worker initialization failed. See" $WORKERLOG " for details."
            echo "To restart Worker, run ./start-spark-standalone-worker-sgx.sh in the docker container."
        fi
    fi
    WORKERPORT=$(netstat -nlp | grep 8081)
    # Default worker port is 8081
    if [ -z "$WORKERPORT" ]; then
        echo "Worker initialization failed. Unable to get worker port at " $WORKERPORT "."
    fi
    
    if [ -n "$WORKERPORT" ] && [ -n "$WORKERSUCCESS" ] ; then
        echo "Worker initialization successful."
    fi
}

all=0
if [ "$#" -lt 1 ]; then
    echo "No argument passed, detecting all component states."
    all=$((all+1))
else
    for arg in "$@"
    do
        if [ "$arg" == all ]; then
            echo "Detecting all component states."
            all=$((all+1))
        fi
    done
fi


if [ "$#" -gt 2 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\""
elif [ "$all" -eq 1 ]; then 
    master
    worker
else 
    for arg in "$@"
    do
        if [ "$arg" == master ]; then
            master
        elif [ "$arg" == worker ]; then
            worker
        else 
            echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\""
        fi
    done
fi
#!/bin/bash
# Acceptable arguments: master, worker, all

MASTERLOG="/ppml/trusted-big-data-ml/spark-master-sgx.log"
WORKERLOG="/ppml/trusted-big-data-ml/spark-worker-sgx.log"

master () {
    echo "(1/2) Detecting master state..."
    MASTERSUCCESS=""
    test -f  $MASTERLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find master log at" $MASTERLOG 
    else 
        MASTERSUCCESS=$(cat $MASTERLOG | grep "I have been elected leader")
        if [ -z "$MASTERSUCCESS" ] ; then
            echo "Master initialization failed. See" $MASTERLOG " for details."
            echo "To restart Master, run ./start-spark-standalone-master-sgx.sh in the docker container."
        fi
    fi
    MASTERPORT=$(netstat -nlp | grep 8080)
    # Default master port is 8080
    if [ -z "$MASTERPORT" ]; then
        echo "Master initialization failed. Unable to get master port at " $MASTERPORT "."
    fi
    
    if [ -n "$MASTERPORT" ] && [ -n "$MASTERSUCCESS" ] ; then
        echo "Master initialization successful."
    fi
}

worker () {
    echo "(2/2) Detecting worker state..."
    WORKERSUCCESS=""
    test -f  $WORKERLOG
    if [ $? -eq 1 ] ; then
        echo "Cannot find worker log at" $WORKERLOG 
    else 
        WORKERSUCCESS=$(cat $WORKERLOG | grep "Successfully registered with master")
        if [ -z "$WORKERSUCCESS" ] ; then
            echo "Worker initialization failed. See" $WORKERLOG " for details."
            echo "To restart Worker, run ./start-spark-standalone-worker-sgx.sh in the docker container."
        fi
    fi
    WORKERPORT=$(netstat -nlp | grep 8081)
    # Default worker port is 8081
    if [ -z "$WORKERPORT" ]; then
        echo "Worker initialization failed. Unable to get worker port at " $WORKERPORT "."
    fi
    
    if [ -n "$WORKERPORT" ] && [ -n "$WORKERSUCCESS" ] ; then
        echo "Worker initialization successful."
    fi
}

all=0
if [ "$#" -lt 1 ]; then
    echo "No argument passed, detecting all component states."
    all=$((all+1))
else
    for arg in "$@"
    do
        if [ "$arg" == all ]; then
            echo "Detecting all component states."
            all=$((all+1))
        fi
    done
fi


if [ "$#" -gt 2 ]; then
    echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\""
elif [ "$all" -eq 1 ]; then 
    master
    worker
else 
    for arg in "$@"
    do
        if [ "$arg" == master ]; then
            master
        elif [ "$arg" == worker ]; then
            worker
        else 
            echo "Acceptable arguments: \"all\", or one or more among \"master\", \"worker\""
        fi
    done
fi
