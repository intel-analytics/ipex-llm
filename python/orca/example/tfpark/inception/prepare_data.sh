#!/usr/bin/env bash

if [[ -z "${ANALYTICS_ZOO_HOME}" ]]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

if [[ -z "${SPARK_HOME}" ]]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

if [[ -z "$1" ]]; then
    echo "Please specify the number of cores of you cluster"
    exit 1
fi


TRAIN_FILE=ILSVRC2012_img_train.tar
VAL_FILE=ILSVRC2012_img_val.tar

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "Please execute this script in the folder where ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar are located "
    exit 1
fi

if [[ ! -f "${VAL_FILE}" ]]; then
    echo "Please execute this script in the folder where ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar are located "
    exit 1
fi

mkdir train
mv $TRAIN_FILE train/
cd train
echo "Extracting training images"
tar -xvf $TRAIN_FILE

mv $TRAIN_FILE ../

find . -name "*.tar" | while read CLASS_NAME ; do mkdir -p "${CLASS_NAME%.tar}"; tar -xvf "${CLASS_NAME}" -C "${CLASS_NAME%.tar}"; done

cd ../
mkdir val
mv $VAL_FILE val/
cd val
echo "Extracting validation images"
tar -xvf $VAL_FILE
cat ../classes.lst | while read CLASS_NAME;
do
    mkdir -p ${CLASS_NAME};
done
cat ../img_class.lst | while read PARAM;
do
mv ${PARAM/ n[0-9]*/} ${PARAM/ILSVRC*JPEG /};
done
mv $VAL_FILE ../

cd ..
rm -f train/*.tar

mkdir sequence
echo "Generating sequence files"
bash $ANALYTICS_ZOO_HOME/bin/spark-shell-with-zoo.sh --driver-memory 20g --class com.intel.analytics.bigdl.models.utils.ImageNetSeqFileGenerator -f . -o ./sequence/ -p $1
