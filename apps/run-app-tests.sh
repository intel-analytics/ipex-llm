#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

RUN_PART1=0
RUN_PART2=0
RUN_PART3=0
if [ $1 = 1 ]; then
RUN_PART1=1
RUN_PART2=0
RUN_PART3=0
elif [ $1 = 2 ]; then
RUN_PART1=0
RUN_PART2=1
RUN_PART3=0
elif [ $1 = 3 ]; then
RUN_PART1=0
RUN_PART2=0
RUN_PART3=1
else
RUN_PART1=1
RUN_PART2=1
RUN_PART3=1
fi

if [ $RUN_PART1 = 1 ]; then
echo "#1 start app test for anomaly-detection-nyc-taxi"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi

chmod +x ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh

${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
sed "s/nb_epoch=30/nb_epoch=15/g" ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py >${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py
now=$(date "+%s")
time1=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py
echo "#1 anomaly-detection-nyc-taxi time used:$time1 seconds"

echo "#2 start app test for object-detection"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection
FILENAME="${ANALYTICS_ZOO_HOME}/apps/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/object-detection/train_dog.mp4"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/train_dog.mp4 -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget https://s3.amazonaws.com/analytics-zoo-data/train_dog.mp4 -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi

FILENAME="/root/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/ffmpeg-linux64-v3.3.1 -P /root/.imageio/ffmpeg/
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py
now=$(date "+%s")
time2=$((now-start))
echo "#2 object-detection time used:$time2 seconds"

echo "#3 start app test for ncf-explicit-feedback"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/ncf-explicit-feedback
sed "s/end_trigger=MaxEpoch(10)/end_trigger=MaxEpoch(5)/g" ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/ncf-explicit-feedback.py >${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py
now=$(date "+%s")
time3=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py
echo "#3 ncf-explicit-feedback time used:$time3 seconds"

echo "#4 start app test for wide_n_deep"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/wide_n_deep
sed "s/end_trigger=MaxEpoch(10)/end_trigger=MaxEpoch(5)/g" ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/wide_n_deep.py >${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py
now=$(date "+%s")
time4=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py
echo "#4 wide_n_deep time used:$time4 seconds"

echo "#5 start app test for using_variational_autoencoder_to_generate_digital_numbers"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py
now=$(date "+%s")
time5=$((now-start))
echo "#5 using_variational_autoencoder_to_generate_digital_numbers time used:$time5 seconds"

echo "#6 start app test for image-similarity"
#timer
start=$(date "+%s")
 ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity
 sed "s%/tmp/images%${ANALYTICS_ZOO_HOME}/apps/image-similarity%g;s%/googlenet_places365/deploy.prototxt%/googlenet_places365/deploy_googlenet_places365.prototxt%g;s%/vgg_16_places365/deploy.prototxt%/vgg_16_places365/deploy_vgg16_places365.prototxt%g;s%./samples%${ANALYTICS_ZOO_HOME}/apps/image-similarity/samples%g" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity.py >${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/imageClassification.tar.gz"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading images"

   wget $FTP_URI/analytics-zoo-data/imageClassification.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/imageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity

   echo "Finished downloading images"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/deploy_googlenet_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 deploy model"

   wget https://analytics-zoo-data.s3.amazonaws.com/deploy_googlenet_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/googlenet_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 weight model"

   wget http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
 FILENAME=" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/deploy_vgg16_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG deploy model"

   wget https://analytics-zoo-data.s3.amazonaws.com/deploy_vgg16_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/vgg16_hybrid1365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG weight model"

   wget $FTP_URI/analytics-zoo-models/image-classification/vgg16_places365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi
 ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
 now=$(date "+%s")
time6=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
echo "#6 image-similarity time used:$time6 seconds"
fi

if [ $RUN_PART2 = 1 ]; then
echo "#7 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model --no-host-directories
   echo "Finished"
fi

FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py,${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
now=$(date "+%s")
time7=$((now-start))
echo "#7 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time7 seconds"

echo "#8 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py,${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
now=$(date "+%s")
time8=$((now-start))
echo "#8 using_variational_autoencoder_to_generate_faces time used:$time8 seconds"
fi

if [ $RUN_PART3 = 1 ]; then
echo "#9 start app test for image-augmentation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-memory 1g  \
        --executor-memory 1g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py
now=$(date "+%s")
time9=$((now-start))
echo "#9 image-augmentation time used:$time9 seconds"

echo "#10 start app test for dogs-vs-cats"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning

sed "s/file:\/\/path\/to\/data\/dogs-vs-cats\/demo/demo/g;s/path\/to\/model\/bigdl_inception-v1_imagenet_0.4.0.model/demo\/bigdl_inception-v1_imagenet_0.4.0.model/g" ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning.py >${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

FILENAME="${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/bigdl_inception-v1_imagenet_0.4.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading model"

   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model -P demo

   echo "Finished downloading model"
fi

FILENAME="${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading dogs and cats images"
   wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip  -P ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/ ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train.zip
   mkdir -p demo/dogs
   mkdir -p demo/cats
   cp ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train/cat.7* demo/cats
   cp ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train/dog.7* demo/dogs
   echo "Finished downloading images"
fi
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

now=$(date "+%s")
time10=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py
echo "#10 dogs-vs-cats time used:$time10 seconds"

echo "#11 start app test for sentiment-analysis"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment

FILENAME="/tmp/.bigdl/dataset/glove.6B.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading glove6B"
   wget -P /tmp/.bigdl/dataset/ $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip
   echo "Finished"
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py
now=$(date "+%s")
time11=$((now-start))
echo "#11 sentiment-analysis time used:$time11 seconds"

echo "#12 start app test for image_classification_inference"
#timer
start=$(date "+%s")
 ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference
 sed "s%/path/to/yourdownload%${ANALYTICS_ZOO_HOME}/apps/tfnet%g;s%file:///path/toyourdownload/dogs-vs-cats/train%${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain%g;s%test.jpg%${ANALYTICS_ZOO_HOME}/apps/tfnet/test.jpg%g;s%imagenet_class_index.json%${ANALYTICS_ZOO_HOME}/apps/tfnet/imagenet_class_index.json%g" ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference.py > ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
 ModelPath="${ANALYTICS_ZOO_HOME}/apps/tfnet/models/"
 rm -rf "$ModelPath"
 echo "Downloading model"
 mkdir -p ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets
 touch ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/__init__.py
 touch ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
 echo "from nets.inception_v1 import inception_v1" >> ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
 echo "from nets.inception_v1 import inception_v1_arg_scope" >> ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
 wget https://analytics-zoo-data.s3.amazonaws.com/inception_utils.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/
 wget https://analytics-zoo-data.s3.amazonaws.com/inception_v1.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/
 echo "Finished downloading model"
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint/inception_v1.ckpt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading inception_v1 checkpoint"
   
   wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint/inception_v1_2016_08_28.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint
   
   echo "Finished downloading checkpoint"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading dogs and cats images"
   
   wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/minitrain.zip -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/data
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain.zip
   #wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/data
   #unzip -d ${ANALYTICS_ZOO_HOME}/apps/tfnet/data ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/train.zip
    echo "Finished downloading images"
fi
 ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
 now=$(date "+%s")
time12=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
echo "#12 image_classification_inference time used:$time12 seconds"

echo "#13 start app test for image-augmentation-3d"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --driver-memory 1g  \
        --executor-memory 1g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d.py
now=$(date "+%s")
time13=$((now-start))
echo "#13 image-augmentation-3d time used:$time13 seconds"

echo "#14 start app test for anomaly-detection-hd"
#timer
start=$(date "+%s")
chmod +x $ANALYTICS_ZOO_HOME/bin/data/HiCS/get_HiCS.sh
$ANALYTICS_ZOO_HOME/bin/data/HiCS/get_HiCS.sh
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
sed -i '127,273d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
now=$(date "+%s")
time14=$((now-start))
echo "#14 anomaly-detection-hd time used:$time14 seconds"

echo "#15 start app test for pytorch face-generation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
sed -i '/plt./d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
now=$(date "+%s")
time15=$((now-start))
echo "#15 pytorch face-generation time used:$time15 seconds"

fi

echo "#1 anomaly-detection-nyc-taxi time used:$time1 seconds"
echo "#2 object-detection time used:$time2 seconds"
echo "#3 ncf-explicit-feedback time used:$time3 seconds"
echo "#4 wide_n_deep time used:$time4 seconds"
echo "#5 using_variational_autoencoder_to_generate_digital_numbers time used:$time5 seconds"
echo "#6 image-similarity time used:$time6 seconds"
echo "#7 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time7 seconds"
echo "#8 using_variational_autoencoder_to_generate_faces time used:$time8 seconds"
echo "#9 image-augmentation time used:$time9 seconds"
echo "#10 dogs-vs-cats time used:$time10 seconds"
echo "#11 sentiment-analysis time used:$time11 seconds"
echo "#12 image_classification_inference time used:$time12 seconds"
echo "#13 image-augmentation-3d time used:$time13 seconds"
echo "#14 anomaly-detection-hd time used:$time14 seconds"
echo "#15 pytorch face-generation time used:$time15 seconds"
