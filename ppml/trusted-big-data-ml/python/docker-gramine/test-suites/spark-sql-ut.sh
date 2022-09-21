#!/bin/bash
mkdir -p /ppml/trusted-big-data-ml/logs/runtime
mkdir -p /ppml/trusted-big-data-ml/logs/reporter

cd /ppml/trusted-big-data-ml

./clean.sh
for suite in `cat /ppml/trusted-big-data-ml/work/test-suites/sparkSqlSuites.txt`
do
    gramine-argv-serializer bash -c "/opt/jdk8/bin/java -cp '$SPARK_HOME/conf/:$SPARK_HOME/jars/*:$SPARK_HOME/test-jars/*:$SPARK_HOME/test-classes/' \
                                    -Xmx8g -Dspark.testing=true -Djdk.lang.Process.launchMechanism=posix_spawn -XX:MaxMetaspaceSize=256m -Dos.name='Linux' \
                                    -Dspark.test.home=/ppml/trusted-big-data-ml/work/spark-3.1.2 -Dspark.python.use.daemon=false -Dspark.python.worker.reuse=false \
                                    -Dspark.driver.host=127.0.0.1 org.scalatest.tools.Runner -s ${suite} -fF /ppml/trusted-big-data-ml/logs/reporter/${suite}.txt" \
                                    > /ppml/trusted-big-data-ml/secured_argvs
    ./init.sh
    gramine-sgx bash 2>&1 | tee /ppml/trusted-big-data-ml/logs/runtime/${suite}.log
    echo "##########$suite Test:"
    if [ -z "$(grep "All tests passed" /ppml/trusted-big-data-ml/logs/reporter/${suite}.txt)" ]
    then
        echo "failed"
        exit 1
    else
        echo "pass"
    fi
    echo "##########$suite Test Done"
done
