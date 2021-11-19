#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
}

'''
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
time=$((now - start))
echo "#1 Total time cost ${time} seconds"

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
time=$((now - start))
echo "#2 Total time cost ${time} seconds"

echo "#3 start test for dllib textclassifier"
#timer
start=$(date "+%s")
#run the example
rm -rf /tmp/news20
${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/news20 /tmp/news20
ls /tmp/news20
python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/textclassifier/textclassifier.py --on-yarn --max_epoch 3 --model cnn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib textclassifier failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#3 Total time cost ${time} seconds"


echo "#4 start test for dllib autograd custom"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/dllib/examples/autograd/custom.py --cluster-mode "yarn-client"
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib autograd custom failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#4 Total time cost ${time} seconds"


echo "#5 start test for dllib autograd customloss"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/dllib/examples/autograd/customloss.py --cluster-mode "yarn-client"
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib autograd customloss failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#5 Total time cost ${time} seconds"


echo "#6 start test for dllib nnframes_imageInference"

if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageInference/ImageInferenceExample.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100 --cluster-mode "yarn-client"
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib nnframes_imageInference failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#6 Total time cost ${time} seconds"


echo "#7 start test for dllib nnframes_imageTransfer learning"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/dogs_cats/samples --nb_epoch 2 --cluster-mode "yarn-client"
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "dllib nnframes_imageTransfer learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#7 Total time cost ${time} seconds"
'''

echo "#9 start test for orca learn/tf/image_segmentation/image_segmentation.py"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
  --batch_size 64 \
  --file_path /bigdl2.0/data/carvana \
  --non_interactive --epochs 1 --cluster-mode yarn-client
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca learn/tf/image_segmentation/image_segmentation.py failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#7 Total time cost ${time} seconds"
