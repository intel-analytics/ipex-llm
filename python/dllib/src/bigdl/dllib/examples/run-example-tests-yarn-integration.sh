#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}


echo "start test for dllib rnn"
echo "start test for dllib custom"
echo "start test for dllib custom loss"
echo "start test for dllib imageframe inception validation"
echo "start test for dllib keras imdb bigdl backend"
echo "start test for dllib keras imdb cnn lstm"
echo "start test for dllib keras mnist cnn"
echo "start test for dllib nnframes image transfer learning"
echo "start test for dllib nnframes image inference"

# echo "#1 start test for dllib lenet5"

# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/mnist
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/mnist /tmp/mnist
# ls /tmp/mnist
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/lenet/lenet5.py --on-yarn -n 1
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib lenet5 failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#1 Total time cost ${time} seconds"

# echo "#2 start test for dllib inception"

# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/inception/inception.py -f ${HDFS_URI}/imagenet-mini \
# 	--batchSize 128 \
# 	--learningRate 0.065 \
# 	--weightDecay 0.0002 \
# 	--executor-memory 20g \
# 	--driver-memory 20g \
# 	--executor-cores 4 \
# 	--num-executors 4 \
#         -i 20
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib inception failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#2 Total time cost ${time} seconds"

# echo "#3 start test for dllib textclassifier"
# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/news20
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/news20 /tmp/news20
# ls /tmp/news20
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/textclassifier/textclassifier.py --on-yarn --max_epoch 3 --model cnn
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib textclassifier failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#3 Total time cost ${time} seconds"

# echo "#4 start test for orca bigdl transformer"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/attention/transformer.py --cluster_mode yarn_cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca transformer failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4 Total time cost ${time} seconds"


echo "#5 start test for orca bigdl imageInference"
#timer
start=$(date "+%s")
if [ -f models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P models
fi
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/bigdl/imageInference/imageInference.py \
  -m models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100 \
  --cluster_mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca imageInference failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#4 Total time cost ${time} seconds"
