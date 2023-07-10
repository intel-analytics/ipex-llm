#!/bin/bash
configFile=""
backend_core_list=""
frontend_core=""
SGX_ENABLED=$SGX_ENABLED
ATTESTATION=$ATTESTATION

usage() {
    echo "Usage: $0 [-c <configfile>] [-b <core# for each backend worker>] [-f <core# for frontend worker>] [-X <enable sgx>] [-A <enable attestation>]"

    echo "ATTENTION: Please allocate at least two dedicated cores for frontend to avoid lock contention in SGX environment"
    echo "The following example command will launch 2 backend workers (as specified in /ppml/torchserve_config)."
    echo "The first backend worker will be pinned to core 0 while the second backend worker will be pinned to core 1."
    echo "The frontend worker will be pinned to core 5,6"
    echo "Example: $0 -c /ppml/torchserve_config -b '0,1' -f '5,6'"
    echo "To launch within SGX environment using gramine"
    echo "Example: $0 -c /ppml/torchserve_config -b '0,1' -f '5,6' -X"
    exit 0
}

while getopts ":b:c:f:XA" opt
do
    case $opt in
        b)
            backend_core_list=$OPTARG
            ;;
        c)
            configFile=$OPTARG
            ;;
        f)
            frontend_core=$OPTARG
            ;;
        X)
            SGX_ENABLED="true"
            ;;
        A)
            ATTESTATION="true"
            ;;
        *)
            echo "Error: unknown positional arguments"
            usage
            ;;
    esac
done

# Check backend_core_list and frontend_core has values
if [ -z "${backend_core_list}" ] || [ -z "${frontend_core}" ]; then
    echo "Error: please specify backend core lists and frontend core"
    usage
fi

# Check config file exists
if [ ! -f "${configFile}" ]; then
    echo "Error: cannot find config file"
    usage
fi

declare -a cores=($(echo "$backend_core_list" | tr "," " "))
declare -a fcores=($(echo "$frontend_core" | tr "," " "))

# In case we change the path name in future
cd /ppml || exit

port=9000
sgx_flag=""

if [[ $SGX_ENABLED == "true" ]]; then
    ./init.sh
    sgx_flag=" -X "
    # Check we allocate at least two cores for frontend
    if [ ${#fcores[@]} -lt "2" ]; then
        echo "Please allocate at least two dedicate cores for frontend when exeuting in SGX env"
        exit
    fi
fi

if [[ $ATTESTATION == "true" ]]; then
    attestation_flag=" -A "
fi

# Consider the situation where we have multiple models, and each load serveral workers.
while read -r line
do
    if [[ $line =~ "minWorkers" ]]; then
        line=${line#*\"minWorkers\": }
        num=${line%%,*}
        line=${line#*,}

        if [ ${#cores[@]} != "$num" ]; then
            echo "Error: worker number does not equal to the length of core list"
            exit 1
        fi

        for ((i=0;i<num;i++,port++))
        do
        (
            bash /ppml/torchserve/start-torchserve-backend.sh -p $port -c "${cores[$i]}" $sgx_flag $attestation_flag
        )&
        done
    fi
done < "$configFile"
(
    bash /ppml/torchserve/start-torchserve-frontend.sh -c "$configFile" -f "$frontend_core" $sgx_flag $attestation_flag
)&
wait

