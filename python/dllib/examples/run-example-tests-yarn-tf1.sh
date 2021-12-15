#!/bin/bash
# echo "#1 start test for dllib lenet5"

# #timer
# start=$(date "+%s")
# #run the example
# rm -rf /tmp/mnist
# ${HADOOP_HOME}/bin/hadoop fs -get ${HDFS_URI}/mnist /tmp/mnist
# ls /tmp/mnist
# python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/lenet/lenet5.py --cluster-mode yarn-client -n 1
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
#       --batchSize 128 \
#       --learningRate 0.065 \
#       --weightDecay 0.0002 \
#       --executor-memory 20g \
#       --driver-memory 20g \
#       --executor-cores 4 \
#       --num-executors 4 \
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


# echo "#4 start test for dllib autograd custom"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/autograd/custom.py --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib autograd custom failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#4 Total time cost ${time} seconds"


# echo "#5 start test for dllib autograd customloss"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/autograd/customloss.py --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib autograd customloss failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#5 Total time cost ${time} seconds"


# echo "#6 start test for dllib nnframes_imageInference"

# if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
#   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
# else
#   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
#     -P analytics-zoo-models
# fi

# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageInference/ImageInferenceExample.py \
#   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
#   -f ${HDFS_URI}/kaggle/train_100 --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib nnframes_imageInference failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#6 Total time cost ${time} seconds"


# echo "#7 start test for dllib nnframes_imageTransfer learning"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/dllib/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
#   -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
#   -f ${HDFS_URI}/dogs_cats/samples --nb_epoch 2 --cluster-mode "yarn-client"
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "dllib nnframes_imageTransfer learning failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#7 Total time cost ${time} seconds"

############################# client
# echo "#8 start test for orca learn transfer_learning"
# #timer 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py --data_dir /data --cluster_mode yarn-client --download False
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
# #   clear_up
#   echo "orca learning transfer_learning failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#8 Total time cost ${time} seconds"

echo "#20 start test for orca tf basic_text_classification basic_text_classification"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py --cluster_mode yarn-client --data_dir /data/imdb
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#20 Total time cost ${time} seconds"

# echo "#21 start test for orca tf image_segmentation image_segmentation.py"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
#   --batch_size 64 \
#   --file_path /data/carvana \
#   --non_interactive --epochs 1 --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca image_segmentation failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#21 Total time cost ${time} seconds"

# echo "#22 start test for orca inception inception"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception.py  \
#   --imagenet ${HDFS_URI}/imagenettfrecord \
#   -b 128 --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca inception failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#22 Total time cost ${time} seconds"

# echo "#9 start test for orca learn tf2 resnet"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py  \
#   --worker_num 2 \
#   --cores 32 \
#   --memory "20g" \
#   --data_dir /data/imagenettfrecord/train \
#   --cluster_mode yarn-client
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca learning learn tf2 resnet failed"
#   exit $exit_status
# fi
# now==$(date "+%s")
# time==$((now - start))
# echo "#9 Total time cost ${time} seconds"


###################### cluster

# echo "#8 start test for orca learn transfer_learning"
# #timer 
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/transfer_learning/transfer_learning.py --data_dir /data --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
# #   clear_up
#   echo "orca learning transfer_learning failed"
# #  exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#8 Total time cost ${time} seconds"

echo "#20 start test for orca tf basic_text_classification basic_text_classification"
#timer
start=$(date "+%s")
#run the example
python ${BIGDL_ROOT}/python/orca/example/learn/tf/basic_text_classification/basic_text_classification.py --cluster_mode yarn-cluster --data_dir /data/imdb
exit_status=$?
if [ $exit_status -ne 0 ]; then
  #clear_up
  echo "orca basic_text_classification failed"
#  exit $exit_status
fi
now=$(date "+%s")
time=$((now - start))
echo "#20 Total time cost ${time} seconds"

# echo "#21 start test for orca tf image_segmentation image_segmentation.py"
# #timer success
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/image_segmentation/image_segmentation.py \
#   --batch_size 64 \
#   --file_path /data/carvana \
#   --non_interactive --epochs 1 --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca image_segmentation failed"
# #  exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#21 Total time cost ${time} seconds"

# echo "#22 start test for orca inception inception"
# #timer ModuleNotFoundError: No module named 'inception_preprocessing'
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf/inception/inception.py  \
#   --imagenet ${HDFS_URI}/imagenettfrecord \
#   -b 128 --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca inception failed"
# #  exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#22 Total time cost ${time} seconds"

# echo "#9 start test for orca learn tf2 resnet"
# #timer  success
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/resnet/resnet-50-imagenet.py  \
#   --worker_num 2 \
#   --cores 32 \
#   --memory "20g" \
#   --data_dir /data/imagenettfrecord/train \
#   --cluster_mode yarn-cluster
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   #clear_up
#   echo "orca learning learn tf2 resnet failed"
#   exit $exit_status
# fi
# now==$(date "+%s")
# time==$((now - start))
# echo "#9 Total time cost ${time} seconds"

###########################

# rm -rf  /data/checkpoints
# mkdir /data/checkpoints
# echo "#23 start test for orca yolov3 yoloV3"
# #timer
# start=$(date "+%s")
# #run the example
# python ${BIGDL_ROOT}/python/orca/example/learn/tf2/yolov3/yoloV3.py  \
#   --data_dir /data/yolov3 \
#   --output_data /data/yolov3/parquet \
#   --weights /data/yolov3/yolov3.weights \
#   --names /data/yolov3/voc2012.names \
#   --epochs 1 --cluster_mode yarn-client \
#   --memory 20g \
#   --object_store_memory 10g \
#   --checkpoint ${HDFS_URI}/data/checkpoints/yolov3.tf \
#   --checkpoint_folder ${HDFS_URI}/data/checkpoints
# exit_status=$?
# if [ $exit_status -ne 0 ]; then
#   clear_up
#   echo "orca yolov3 failed"
#   exit $exit_status
# fi
# now=$(date "+%s")
# time=$((now - start))
# echo "#23 Total time cost ${time} seconds"


