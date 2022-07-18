#!/usr/bin/env bash

# On Ubuntu, this script requires apache2-utils for the ab binary.
#
# Run like: ./benchmark-http.sh host:port
#
# It also works with HTTPS, e.g., ./benchmark-http.sh https://localhost:8443

declare -A THROUGHPUTS
declare -A LATENCIES
LOOP=${LOOP:-1}
DOWNLOAD_HOST=$1
DOWNLOAD_FILE=random/10K.1.html
REQUESTS=10000
CONCURRENCY_LIST=${CONCURRENCY_LIST:-"1 2 4 8 16 32 64 128 256"}
OPTIONS="-k"
RESULT=result-$(date +%y%m%d-%H%M%S)

touch "$RESULT"

RUN=0
while [ $RUN -lt "$LOOP" ]
do
    for CONCURRENCY in $CONCURRENCY_LIST
    do
        rm -f OUTPUT
        echo "ab $OPTIONS -n $REQUESTS -c $CONCURRENCY $DOWNLOAD_HOST/$DOWNLOAD_FILE"
        ab $OPTIONS -n $REQUESTS -c "$CONCURRENCY" "$DOWNLOAD_HOST/$DOWNLOAD_FILE" > OUTPUT || exit $?

        sleep 5

        THROUGHPUT=$(grep -m1 "Requests per second:" OUTPUT | awk '{ print $4 }')
        LATENCY=$(grep -m1 "Time per request:" OUTPUT | awk '{ print $4 }')
        FAILED=$(grep -m1 "Failed requests:" OUTPUT | awk '{print $3 }')
        THROUGHPUTS[$CONCURRENCY]="${THROUGHPUTS[$CONCURRENCY]} $THROUGHPUT"
        LATENCIES[$CONCURRENCY]="${LATENCIES[$CONCURRENCY]} $LATENCY"
        echo "concurrency=$CONCURRENCY, throughput=$THROUGHPUT, latency=$LATENCY, failed=$FAILED"
    done
    (( RUN++ ))
done

for CONCURRENCY in $CONCURRENCY_LIST
do
    THROUGHPUT=$(echo "${THROUGHPUTS[$CONCURRENCY]}" | tr " " "\n" | sort -n | awk '{a[NR]=$0}END{if(NR%2==1)print a[int(NR/2)+1];else print(a[NR/2-1]+a[NR/2])/2}')
    LATENCY=$(echo "${LATENCIES[$CONCURRENCY]}" | tr " " "\n" | sort -n | awk '{a[NR]=$0}END{if(NR%2==1)print a[int(NR/2)+1];else print(a[NR/2-1]+a[NR/2])/2}')
    echo "$THROUGHPUT,$LATENCY" >> "$RESULT"
done

echo "Result file: $RESULT"
