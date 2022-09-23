#!/bin/bash
status_3_local_spark_pi=1
status_4_local_spark_wordcount=1

if [ $status_3_local_spark_pi -ne 0 ]; then
echo "example.3 local spark, pi"
cd /ppml/trusted-big-data-ml
./clean.sh
/graphene/Tools/argv_serializer bash -c "/opt/jdk8/bin/java \
   -cp '/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*' \
   -Xmx1g org.apache.spark.deploy.SparkSubmit \
   --master 'local[4]' \
   --conf spark.python.use.daemon=false \
   --conf spark.python.worker.reuse=false \
   /ppml/trusted-big-data-ml/work/spark-3.1.3/examples/src/main/python/pi.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-pi-sgx.log && \
   cat test-pi-sgx.log | egrep 'roughly'
status_3_local_spark_pi=$(echo $?)
fi


if [ $status_4_local_spark_wordcount -ne 0 ]; then
echo "example.4 local spark, test-wordcount"
cd /ppml/trusted-big-data-ml
./clean.sh
/graphene/Tools/argv_serializer bash -c "export PYSPARK_PYTHON=/usr/bin/python && /opt/jdk8/bin/java \
   -cp '/ppml/trusted-big-data-ml/work/spark-3.1.3/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.3/jars/*' \
   -Xmx1g org.apache.spark.deploy.SparkSubmit \
   --master 'local[4]' \
   --conf spark.python.use.daemon=false \
   --conf spark.python.worker.reuse=false \
   /ppml/trusted-big-data-ml/work/spark-3.1.3/examples/src/main/python/wordcount.py \
   /ppml/trusted-big-data-ml/work/examples/helloworld.py" > /ppml/trusted-big-data-ml/secured-argvs
./init.sh
SGX=1 ./pal_loader bash 2>&1 | tee test-wordcount-sgx.log && \
   cat test-wordcount-sgx.log | egrep 'print'
status_4_local_spark_wordcount=$(echo $?)
fi

echo "#### example.3 Excepted result(Pi): Pi is roughly XXX"
echo "---- example.3 Actual result: "
cat test-pi-sgx.log | egrep 'roughly'

echo -e "#### example.4 Excepted result(WordCount): \nXXX1: 1\nXXX2: 2\nXXX3: 3"
echo "---- example.4 Actual result: "
cat test-wordcount-sgx.log | egrep -a 'import.*: [0-9]*$'
