#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}


"#3 start test for dllib textclassifier"
"start test for dllib rnn"
"start test for dllib custom"
"start test for dllib custom loss"
"start test for dllib imageframe inception validation"
"start test for dllib keras imdb bigdl backend"
"start test for dllib keras imdb cnn lstm"
"start test for dllib keras mnist cnn"
"start test for dllib nnframes image transfer learning"
"start test for dllib nnframes image inference"

echo "#1 start test for dllib lenet5"

#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/mnist
${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/mnist /tmp/mnist
ls /tmp/mnist
python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/lenet/lenet5.py --on-yarn -n 1
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib lenet5 failed"
  exit $exit_status
fi
now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for dllib inception"

#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/inception/inception.py -f ${HDFS_URI}/imagenet-mini \
	--batchSize 128 \
	--learningRate 0.065 \
	--weightDecay 0.0002 \
	--executor-memory 20g \
	--driver-memory 20g \
	--executor-cores 4 \
	--num-executors 4 \
        -i 20
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib inception failed"
  exit $exit_status
fi
now=$(date "+%s")
time2=$((now - start))

