#!/bin/bash

# python files will be submitted as jobs via spark.
cd /ppml/trusted-big-data-ml
mkdir -p /ppml/trusted-big-data-ml/logs/pyspark/sql

# Source Suites Path
SOURCE_SUITES_FILE_PATH="/ppml/trusted-big-data-ml/work/data/Spark-UT-Suites/pysparkSuites"

# temp input data for and result data for ut test
INPUT_SUITES_FILE_PATH="/ppml/trusted-big-data-ml/pysparkInputSuites"
SUCCESS_SUITES_FILE_PATH="/ppml/trusted-big-data-ml/pysparkSucceessSuites"
FAILED_SUITES_FILE_PATH="/ppml/trusted-big-data-ml/pysparkFailedSuites"

# copy source suites to failed suites, assume
cp $SOURCE_SUITES_FILE_PATH $FAILED_SUITES_FILE_PATH

# total test suites
total_test_suites=`cat $SOURCE_SUITES_FILE_PATH | wc -l`

# Run 3 times for spark sql ut
TIMES=3

for ((i=1; i<=TIMES; i++))
do
        # prepare input suites and test failed suites again
        mv $FAILED_SUITES_FILE_PATH $INPUT_SUITES_FILE_PATH
        touch $FAILED_SUITES_FILE_PATH

        while read suite
        do
            # Gramine SGX Command
            export sgx_command="/opt/jdk8/bin/java -cp \
                   /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/conf/:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/jars/*:/ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/examples/jars/* \
                   -Xmx1g org.apache.spark.deploy.SparkSubmit \
                   --master local[4] \
                   --conf spark.network.timeout=10000000 \
                   --conf spark.executor.heartbeatInterval=10000000 \
                   --conf spark.python.use.daemon=false \
                   --conf spark.python.worker.reuse=false \
                   /ppml/trusted-big-data-ml/work/spark-$SPARK_VERSION/python/pyspark/sql/tests/$suite"
            gramine-sgx bash > /ppml/trusted-big-data-ml/logs/pyspark/sql/$suite.log 2>&1

            # Records the number of successful test files number and path.
            if [ -n "$(grep "FAILED" /ppml/trusted-big-data-ml/logs/pyspark/sql/$suite.log -H -o)" ]
            then
                echo "$suite" >> $FAILED_SUITES_FILE_PATH
            else
                echo "$suite" >> $SUCCESS_SUITES_FILE_PATH
            fi
        done < $INPUT_SUITES_FILE_PATH

        # always print the result on the console per time
        echo "Total Test Suites Count: $total_test_suites"
        echo -e "The Success Test Files Count: `cat $SUCCESS_SUITES_FILE_PATH | wc -l`, Below The File List:\n"
        cat $SUCCESS_SUITES_FILE_PATH
        echo -e "\n------------------------------------------------------------------------------------------"
        echo -e "The Failed Test Files Count: `cat $FAILED_SUITES_FILE_PATH | wc -l`, Below The File List:\n"
        cat $FAILED_SUITES_FILE_PATH
	echo -e "\n\n\n"

	if [ `cat $FAILED_SUITES_FILE_PATH | wc -l` -eq 0 ]
        then
	    break
	fi
done
