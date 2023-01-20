#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling bigdl"
    pip uninstall -y bigdl-dllib
    pip uninstall -y bigdl-orca
}

chmod +x ${BIGDL_ROOT}/apps/ipynb2py.sh
RUN_PART1=0
RUN_PART2=0

if [ $1 = 1 ]; then
	RUN_PART1=1
	RUN_PART2=0
elif [ $1 = 2 ]; then
	RUN_PART1=0
	RUN_PART2=1
fi

set -e

if [ $RUN_PART1 = 1 ]; then
echo "#1 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${BIGDL_ROOT}/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${BIGDL_ROOT}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model --no-host-directories
   echo "Finished"
fi
 FILENAME="${BIGDL_ROOT}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${BIGDL_ROOT}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${BIGDL_ROOT}/apps/variational-autoencoder/ ${BIGDL_ROOT}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
python ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
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


${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers

sed "s/nb_epoch = 6/nb_epoch=2/g; s/batch_size=batch_size/batch_size=1008/g" ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py > ${BIGDL_ROOT}/apps/variational-autoencoder/tmp_test.py

export SPARK_DRIVER_MEMORY=12g
#python ${BIGDL_ROOT}/apps/variational-autoencoder/tmp_test.py

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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/anomaly-detection/anomaly-detection-nyc-taxi
wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ${BIGDL_ROOT}/bin/data/NAB/nyc_taxi/
sed "s/nb_epoch=20/nb_epoch=2/g; s/batch_size=1024/batch_size=1008/g" ${BIGDL_ROOT}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py > ${BIGDL_ROOT}/apps/anomaly-detection/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
export BIGDL_ROOT=${BIGDL_ROOT}
python ${BIGDL_ROOT}/apps/anomaly-detection/tmp_test.py

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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/image-similarity/image-similarity
sed "s/setBatchSize(20)/setBatchSize(56)/g;s/setMaxEpoch(2)/setMaxEpoch(1)/g;s%/tmp/images%${BIGDL_ROOT}/apps/image-similarity%g;s%imageClassification%miniimageClassification%g;s%/googlenet_places365/deploy.prototxt%/googlenet_places365/deploy_googlenet_places365.prototxt%g;s%/vgg_16_places365/deploy.prototxt%/vgg_16_places365/deploy_vgg16_places365.prototxt%g;s%./samples%${BIGDL_ROOT}/apps/image-similarity/samples%g" ${BIGDL_ROOT}/apps/image-similarity/image-similarity.py >${BIGDL_ROOT}/apps/image-similarity/tmp.py
FILENAME="${BIGDL_ROOT}/apps/image-similarity/miniimageClassification.tar.gz"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   tar -zxvf ${BIGDL_ROOT}/apps/image-similarity/miniimageClassification.tar.gz -C ${BIGDL_ROOT}/apps/image-similarity
else
   echo "Downloading images"

   wget $FTP_URI/analytics-zoo-data/miniimageClassification.tar.gz -P ${BIGDL_ROOT}/apps/image-similarity
   tar -zxvf ${BIGDL_ROOT}/apps/image-similarity/miniimageClassification.tar.gz -C ${BIGDL_ROOT}/apps/image-similarity

   echo "Finished downloading images"
fi
FILENAME="${BIGDL_ROOT}/apps/image-similarity/googlenet_places365/deploy_googlenet_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_googlenet_places365.prototxt -P ${BIGDL_ROOT}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${BIGDL_ROOT}/apps/image-similarity/googlenet_places365/googlenet_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 weight model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/googlenet_places365.caffemodel -P ${BIGDL_ROOT}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${BIGDL_ROOT}/apps/image-similarity/vgg_16_places365/deploy_vgg16_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_vgg16_places365.prototxt -P ${BIGDL_ROOT}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi
FILENAME="${BIGDL_ROOT}/apps/image-similarity/vgg_16_places365/vgg16_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG weight model"

   wget $FTP_URI/analytics-zoo-models/image-classification/vgg16_places365.caffemodel  -P ${BIGDL_ROOT}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${BIGDL_ROOT}/apps/image-similarity/tmp.py

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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/image-augmentation/image-augmentation

# Run the example
export SPARK_DRIVER_MEMORY=1g
cd ${BIGDL_ROOT}/apps/image-augmentation/
python image-augmentation.py

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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/dogs-vs-cats/transfer-learning

sed "s/setBatchSize(40)/setBatchSize(56)/g; s/file:\/\/path\/to\/data\/dogs-vs-cats\/demo/demo/g;s/path\/to\/model\/bigdl_inception-v1_imagenet_0.4.0.model/demo\/bigdl_inception-v1_imagenet_0.4.0.model/g" ${BIGDL_ROOT}/apps/dogs-vs-cats/transfer-learning.py >${BIGDL_ROOT}/apps/dogs-vs-cats/tmp.py

FILENAME="${BIGDL_ROOT}/apps/dogs-vs-cats/bigdl_inception-v1_imagenet_0.4.0.model"
if [ -f "$FILENAME" ]
 then
    echo "$FILENAME already exists."
 else
    echo "Downloading model"

    wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model -P demo

    echo "Finished downloading model"
 fi

FILENAME="${BIGDL_ROOT}/apps/dogs-vs-cats/train.zip"
if [ -f "$FILENAME" ]
 then
    echo "$FILENAME already exists."
 else
    echo "Downloading dogs and cats images"
    wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip  -P ${BIGDL_ROOT}/apps/dogs-vs-cats
    unzip -d ${BIGDL_ROOT}/apps/dogs-vs-cats/ ${BIGDL_ROOT}/apps/dogs-vs-cats/train.zip
    mkdir -p demo/dogs
    mkdir -p demo/cats
    cp ${BIGDL_ROOT}/apps/dogs-vs-cats/train/cat.7* demo/cats
    cp ${BIGDL_ROOT}/apps/dogs-vs-cats/train/dog.7* demo/dogs
    echo "Finished downloading images"
 fi

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${BIGDL_ROOT}/apps/dogs-vs-cats/tmp.py

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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/image-augmentation-3d/image-augmentation-3d

# Run the example
export SPARK_DRIVER_MEMORY=1g
cd ${BIGDL_ROOT}/apps/image-augmentation-3d/
python image-augmentation-3d.py

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

fi

if [ $RUN_PART2 = 1 ]; then
echo "#8 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")
 ${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
FILENAME="${BIGDL_ROOT}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${BIGDL_ROOT}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${BIGDL_ROOT}/apps/variational-autoencoder/ ${BIGDL_ROOT}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
python ${BIGDL_ROOT}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
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
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/sentiment-analysis/sentiment
sed "s/batch_size = 64/batch_size = 84/g" ${BIGDL_ROOT}/apps/sentiment-analysis/sentiment.py >${BIGDL_ROOT}/apps/sentiment-analysis/tmp_test.py
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
python ${BIGDL_ROOT}/apps/sentiment-analysis/tmp_test.py

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
FILENAME="${BIGDL_ROOT}/apps/anomaly-detection-hd/realworld.zip"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/HiCS/realworld.zip  -P ${BIGDL_ROOT}/apps/anomaly-detection-hd
fi
dataPath="${BIGDL_ROOT}/bin/data/HiCS/"
rm -rf "$dataPath"
mkdir -p ${BIGDL_ROOT}/bin/data/HiCS/
unzip ${BIGDL_ROOT}/apps/anomaly-detection-hd/realworld.zip -d ${BIGDL_ROOT}/bin/data/HiCS/  
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/anomaly-detection-hd/autoencoder-zoo
sed -i '/get_ipython()/d' ${BIGDL_ROOT}/apps/anomaly-detection-hd/autoencoder-zoo.py
sed -i '127,273d' ${BIGDL_ROOT}/apps/anomaly-detection-hd/autoencoder-zoo.py
python ${BIGDL_ROOT}/apps/anomaly-detection-hd/autoencoder-zoo.py
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

echo "#11 start app test for ray paramater-server"
#timer
start=$(date "+%s")

${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/ray/parameter_server/sharded_parameter_server
python ${BIGDL_ROOT}/apps/ray/parameter_server/sharded_parameter_server.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "ray paramater-server failed"
    exit $exit_status
fi
now=$(date "+%s")
time11=$((now-start))

echo "#11 ray paramater-server time used:$time11 seconds"

echo "#12 start app test for image_classification_inference"
#timer
start=$(date "+%s")
${BIGDL_ROOT}/apps/ipynb2py.sh ${BIGDL_ROOT}/apps/tfnet/image_classification_inference

sed "s%/path/to/yourdownload%${BIGDL_ROOT}/apps/tfnet%g;s%file:///path/toyourdownload/dogs-vs-cats/train%${BIGDL_ROOT}/apps/tfnet/data/minitrain%g;s%test.jpg%${BIGDL_ROOT}/apps/tfnet/test.jpg%g;s%imagenet_class_index.json%${BIGDL_ROOT}/apps/tfnet/imagenet_class_index.json%g; s/setBatchSize(16)/setBatchSize(56)/g;" ${BIGDL_ROOT}/apps/tfnet/image_classification_inference.py > ${BIGDL_ROOT}/apps/tfnet/tmp.py
FILENAME="${BIGDL_ROOT}/apps/tfnet/models/*"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading model"

    mkdir -p ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets
    touch ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/__init__.py
    touch ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/inception.py
    echo "from nets.inception_v1 import inception_v1" >> ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/inception.py
    echo "from nets.inception_v1 import inception_v1_arg_scope" >> ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/inception.py
    wget $FTP_URI/analytics-zoo-models/image-classification/inception_utils.py -P ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/
    wget $FTP_URI/analytics-zoo-models/image-classification/inception_v1.py -P ${BIGDL_ROOT}/apps/tfnet/models/research/slim/nets/

   echo "Finished downloading model"
fi
FILENAME="${BIGDL_ROOT}/apps/tfnet/checkpoint/inception_v1.ckpt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading inception_v1 checkpoint"

   wget $FTP_URI/analytics-zoo-models/image-classification/inception_v1_2016_08_28.tar.gz -P ${BIGDL_ROOT}/apps/tfnet/checkpoint
   tar -zxvf ${BIGDL_ROOT}/apps/tfnet/checkpoint/inception_v1_2016_08_28.tar.gz -C ${BIGDL_ROOT}/apps/tfnet/checkpoint

   echo "Finished downloading checkpoint"
fi
FILENAME="${BIGDL_ROOT}/apps/tfnet/data/minitrain.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading dogs and cats images"

   wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/minitrain.zip -P ${BIGDL_ROOT}/apps/tfnet/data
   unzip -d ${BIGDL_ROOT}/apps/tfnet/data/minitrain ${BIGDL_ROOT}/apps/tfnet/data/minitrain.zip
    echo "Finished downloading images"
fi

export SPARK_DRIVER_MEMORY=12g
python ${BIGDL_ROOT}/apps/tfnet/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image_classification_inference failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time12=$((now-start))
rm ${BIGDL_ROOT}/apps/tfnet/tmp.py
echo "#12 image_classification_inference time used:$time12 seconds"

fi
