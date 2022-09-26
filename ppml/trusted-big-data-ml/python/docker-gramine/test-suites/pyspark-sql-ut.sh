#!/bin/bash
mkdir -p /ppml/trusted-big-data-ml/logs/pyspark/sql

cd /ppml/trusted-big-data-ml

./clean.sh
for suite in `cat /ppml/trusted-big-data-ml/work/test-suites/pysparkSqlSuites.txt`
do
    while true
        gramine-argv-serializer bash -c "/opt/jdk8/bin/java -cp \
                                        '/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
                                        -Xmx1g org.apache.spark.deploy.SparkSubmit --master 'local[4]' --conf spark.network.timeout=10000000 \
                                        --conf spark.executor.heartbeatInterval=10000000 --conf spark.python.use.daemon=false \
                                        --conf spark.python.worker.reuse=false \
                                        /ppml/trusted-big-data-ml/work/spark-3.1.2/python/pyspark/sql/tests/$suite" \
                                        > secured_argvs
        ./init.sh
        gramine-sgx bash 2>&1 | tee /ppml/trusted-big-data-ml/logs/pyspark/sql/$suite.log
        echo "##########$suite Test:"
        if [ -n "$(grep "FAILED" /ppml/trusted-big-data-ml/logs/pyspark/sql/$suite.log -H -o)" ]
        then
            echo "failed"
            rm /ppml/trusted-big-data-ml/logs/pyspark/sql/$suite.log
        else
            echo "pass"
            break
        fi
    done
    echo "##########$suite Test Done"
done