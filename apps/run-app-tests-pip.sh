#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y bigdl-dllib
    pip uninstall -y bigdl-orca
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

echo "#1 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model --no-host-directories
   echo "Finished"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
 exit_status=$?

if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_and_deep_feature_loss_to_generate_faces failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "#1 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time1 seconds"

echo "#2 start app test for using_variational_autoencoder_to_generate_digital_numbers"
#timer
start=$(date "+%s")

# get mnist
if [[ ! -z "${FTP_URI}" ]]; then
    if [[ -d /tmp/mnist/ ]]; then
        rm -rf /tmp/datasets/MNIST/
    fi
    wget  $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P /tmp/mnist
    wget  $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P /tmp/mnist
    wget  $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P /tmp/mnist
    wget  $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P /tmp/mnist
fi


${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers

sed "s/nb_epoch = 6/nb_epoch=2/g; s/batch_size=batch_size/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py > ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/tmp_test.py

export SPARK_DRIVER_MEMORY=12g
#python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_to_generate_digital_numbers failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "#2 using_variational_autoencoder_to_generate_digital_numbers time used:$time2 seconds"

echo "#3 start app test for anomaly-detection-nyc-taxi"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi
wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/nyc_taxi.csv
sed "s/nb_epoch=20/nb_epoch=2/g; s/batch_size=1024/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py > ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "#3 anomaly-detection-nyc-taxi time used:$time3 seconds"

echo "#4 start app test for image-similarity"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity
sed "s/setBatchSize(20)/setBatchSize(56)/g;s/setMaxEpoch(2)/setMaxEpoch(1)/g;s%/tmp/images%${ANALYTICS_ZOO_HOME}/apps/image-similarity%g;s%imageClassification%miniimageClassification%g;s%/googlenet_places365/deploy.prototxt%/googlenet_places365/deploy_googlenet_places365.prototxt%g;s%/vgg_16_places365/deploy.prototxt%/vgg_16_places365/deploy_vgg16_places365.prototxt%g;s%./samples%${ANALYTICS_ZOO_HOME}/apps/image-similarity/samples%g" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity.py >${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity
else
   echo "Downloading images"

   wget $FTP_URI/analytics-zoo-data/miniimageClassification.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity

   echo "Finished downloading images"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/deploy_googlenet_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_googlenet_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/googlenet_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 weight model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/googlenet_places365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/deploy_vgg16_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_vgg16_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/vgg16_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG weight model"

   wget $FTP_URI/analytics-zoo-models/image-classification/vgg16_places365.caffemodel  -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-similarity failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time4=$((now-start))
echo "#4 image-similarity time used:$time4 seconds"

echo "#5 start app test for image-augmentation"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-augmentation failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))
echo "#5 image-augmentation time used:$time5 seconds"

echo "#6 start app test for dogs-vs-cats"
start=$(date "+%s")

# Conversion to py file and data preparation

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning

sed "s/setBatchSize(40)/setBatchSize(56)/g; s/file:\/\/path\/to\/data\/dogs-vs-cats\/demo/demo/g;s/path\/to\/model\/bigdl_inception-v1_imagenet_0.4.0.model/demo\/bigdl_inception-v1_imagenet_0.4.0.model/g" ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning.py >${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

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

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "dogs-vs-cats failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time6=$((now-start))
echo "#6 dogs-vs-cats time used:$time6 seconds"

echo "#7 start app test for image-augmentation-3d"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-augmentation-3d failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time7=$((now-start))
echo "#7 image-augmentation-3d time used:$time7 seconds"

echo "#8 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")
 ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
 exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_to_generate_faces failed"
    exit $exit_status
fi
 unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time8=$((now-start))
echo "#8 using_variational_autoencoder_to_generate_faces time used:$time8 seconds"

echo "#9 start app test for sentiment-analysis"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment
sed "s/batch_size = 64/batch_size = 84/g" ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py >${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/tmp_test.py
FILENAME="/tmp/.bigdl/dataset/glove.6B.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading glove6B"
   wget -P /tmp/.bigdl/dataset/ $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip
   echo "Finished"
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "sentiment-analysis failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time9=$((now-start))
echo "#9 sentiment-analysis time used:$time9 seconds"

echo "#10 start app test for anomaly-detection-hd"
#timer
start=$(date "+%s")
FILENAME="${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/realworld.zip"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/HiCS/realworld.zip  -P ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd
fi
dataPath="${ANALYTICS_ZOO_HOME}/bin/data/HiCS/"
rm -rf "$dataPath"
unzip -d ${ANALYTICS_ZOO_HOME}/bin/data/HiCS/  ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/realworld.zip
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
sed -i '127,273d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection-hd failed"
    exit $exit_status
fi
now=$(date "+%s")
time10=$((now-start))
echo "#10 anomaly-detection-hd time used:$time10 seconds"
