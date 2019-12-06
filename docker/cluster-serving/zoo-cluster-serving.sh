#!/bin/bash

# --------------config

/opt/work/redis-5.0.5/src/redis-server --port $REDIS_PORT > /opt/work/redis.log &
echo "redis server started, please check log in /opt/work/redis.log" &
sleep 1

# sleep for 1 sec to ensure server is ready and client could connect
/opt/work/redis-5.0.5/src/redis-cli config set stop-writes-on-bgsave-error no
/opt/work/redis-5.0.5/src/redis-cli config set save ""

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}

parse_yaml config.yaml
eval $(parse_yaml config.yaml)

if [ -z "${spark_master}" ]; then
    echo "master of spark cluster not set, using default value local[*]"
    spark_master=local[*]
fi
if [ -z "${spark_driver_memory}" ]; then
    echo "spark driver memory not set, using default value 4g"
    spark_driver_memory=4g
fi
if [ -z "${spark_executor_memory}" ]; then
    echo "spark executor memory not set, using default value 1g"
    spark_executor_memory=1g
fi
if [ -z "${spark_num_executors}" ]; then
    echo "spark num-executors not set, using default value 1"
    spark_num_executors=1
fi
if [ -z "${spark_executor_cores}" ]; then
    echo "spark executor-cores not set, using default value 4"
    spark_executor_cores=4
fi
if [ -z "${spark_total_executor_cores}" ]; then
    echo "spark executor-cores not set, using default value 4"
    spark_total_executor_cores=4
fi

if [ -z "${params_engine_type}" ]; then    
    params_engine_type=mklblas
fi


${SPARK_HOME}/bin/spark-submit --master ${spark_master} --driver-memory ${spark_driver_memory} --executor-memory ${spark_executor_memory} --num-executors ${spark_num_executors} --executor-cores ${spark_executor_cores} --total-executor-cores ${spark_total_executor_cores} --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=${params_engine_type}" --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=${params_engine_type}" --jars ${SPARK_REDIS_JAR} --class com.intel.analytics.zoo.serving.ClusterServing ${ZOO_JAR}

