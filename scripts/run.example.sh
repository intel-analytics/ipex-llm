#!/bin/bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

BIGDL_VERSION=0.13.0-SNAPSHOT

SPARK1_DIR=spark-1.6.3-bin-hadoop2.6
SPARK1_LINK=https://www.apache.org/dist/spark/spark-1.6.3/$SPARK1_DIR.tgz
SPARK2_DIR=spark-2.0.2-bin-hadoop2.7
SPARK2_LINK=https://www.apache.org/dist/spark/spark-2.0.2/$SPARK2_DIR.tgz
SPARK_DIR=$SPARK1_DIR
SPARK_LINK=$SPARK1_LINK
SPARK_SUBMIT=./$SPARK_DIR/bin/spark-submit
CURRENT=`pwd`
BIGDL2_JAR=$HOME/.m2/repository/com/intel/analytics/bigdl/bigdl-SPARK_2.0/${BIGDL_VERSION}/bigdl-SPARK_2.0-${BIGDL_VERSION}-jar-with-dependencies.jar
BIGDL1_JAR=$HOME/.m2/repository/com/intel/analytics/bigdl/bigdl/${BIGDL_VERSION}/bigdl-${BIGDL_VERSION}-jar-with-dependencies.jar
BIGDL_JAR=$BIGDL1_JAR
DATA_DIR_MNIST=mnist
DATA_DIR_CIFAR=cifar-10-batches-bin
DATA_DIR_IMAGENET=imagenet
MODEL_DIR=model
LEARNING_RATE=0.01
MAX_EPOCH=90
ME=`basename "$0"`

options=$(getopt -o p:m:c:s:o:r:n:b:t:l:f:e:j:h -l spark:,model:,class:,spark-url:,cores:,memory:,nodes:,batch-size:,trained-model:,learning-rate:,hdfs-data-dir:,max-epoch:,bigdl-jar:,help -- "$@")

eval set -- "$options"

while true; do
	case $1 in
		-p|--spark)
			if [ "$2" == "spark_2.x" ]; then
				SPARK_DIR=$SPARK2_DIR
				SPARK_LINK=$SPARK2_LINK
				BIGDL_JAR=$BIGDL2_JAR
				SPARK_SUBMIT=./$SPARK_DIR/bin/spark-submit
			elif [ "$2" == "spark_buildIn" ]; then
				SPARK_SUBMIT=spark-submit	
			fi
			shift 2 ;;
		-m|--model) MODEL=$2; shift 2 ;;
		-c|--class) CLASS=$2; shift 2 ;;
		-s|--spark-url) SPARK_URL="--master $2"; shift 2 ;;
		-o|--cores) CORES=$2; shift 2 ;;
		-r|--memory) MEMORY=$2; shift 2 ;;
		-n|--nodes) NODES=$2; shift 2 ;;
		-b|--batch-size) BATCH_SIZE=$2; shift 2 ;;
		-t|--trained-model) TRAINED_MODEL=$2; shift 2 ;;
		-l|--learning-rate) LEARNING_RATE=$2; shift 2 ;;
		-f|--hdfs-data-dir) HDFS_DATA_DIR=$2; shift 2 ;;
		-e|--max-epoch) MAX_EPOCH=$2; shift 2 ;;
		-j|--bigdl-jar) BIGDL_JAR=$2; shift 2 ;;
		-h|--help)
			echo "Example:"
			echo "train lenet: $ME --model lenet --spark-url spark://10.0.0.1:7077 --cores 32 --memory 200g --nodes 4 --batch-size 512"
			echo "train vgg: $ME --model vgg --spark-url spark://10.0.0.1:7077 --cores 32 --memory 200g --nodes 4 --batch-size 512"
			echo "train inception-v1: $ME --model inception-v1 --spark-url spark://10.0.0.1:7077 --cores 32 --memory 200g --nodes 16 --batch-size 2048 --learning-rate 0.0898 --hdfs-data-dir hdfs://10.0.0.1:9000/imagenet"
			echo "run performance: $ME --model perf --spark-url spark://10.0.0.1:7077 --cores 32 --memory 200g --nodes 4"
			shift; exit 0;
			;;
		--) shift; break ;;
	esac
done
# echo $SPARK_SUBMIT
if [ $SPARK_SUBMIT == "spark-submit" ]; then
	echo "Using build in spark"
elif [ -d $SPARK_DIR ]; then
	echo "Using existing spark dir $SPARK_DIR"
else
	echo "Downloading $SPARK_DIR from $SPARK_LINK ..."
	wget $SPARK_LINK && tar -xzf $SPARK_DIR.tgz
fi

if [ ! -f $BIGDL_JAR ]; then
	if [ "$BIGDL_JAR" == "$BIGDL2_JAR" ]; then
		mvn dependency:get -DremoteRepositories=https://oss.sonatype.org/content/groups/public/ -DgroupId=com.intel.analytics.bigdl -DartifactId=bigdl-SPARK_2.0 -Dversion=$BIGDL_VERSION -Dclassifier=jar-with-dependencies -Dtransitive=false
	else
		mvn dependency:get -DremoteRepositories=https://oss.sonatype.org/content/groups/public/ -DgroupId=com.intel.analytics.bigdl -DartifactId=bigdl -Dversion=$BIGDL_VERSION -Dclassifier=jar-with-dependencies -Dtransitive=false
	fi
fi

[[ ! $MODEL =~ lenet|vgg|inception-v1|perf ]] && {
	echo "ERROR: model must be one of lenet, vgg, inception-v1 or perf"
	exit 1
}

if [ "$MODEL" == "lenet" ]; then
	MODEL_DIR=lenet_model
	DATA_DIR=$DATA_DIR_MNIST
	if [ -d $DATA_DIR ]; then
		echo "Using existing data dir $DATA_DIR"
	else
		echo "Downloading mnist from http://yann.lecun.com/exdb/mnist/ ..."
		mkdir $DATA_DIR
		cd $DATA_DIR
		wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz  && gunzip train-images-idx3-ubyte.gz
		wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz && gunzip train-labels-idx1-ubyte.gz
		wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz && gunzip t10k-images-idx3-ubyte.gz
		wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz && gunzip t10k-labels-idx1-ubyte.gz
		cd ../
	fi
elif [ "$MODEL" == "vgg" ]; then
	MODEL_DIR=vgg_model
	DATA_DIR=$DATA_DIR_CIFAR
	if [ -d $DATA_DIR ]; then
		echo "Using existing data dir $DATA_DIR"
	else
		echo "Downloading mnist from https://www.cs.toronto.edu/~kriz/cifar.html ..."
		wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz && tar -xzf cifar-10-binary.tar.gz
	fi
elif [ "$MODEL" == "inception-v1" ]; then
	MODEL_DIR=inception_model
	# check batch_size
	if [ "$BATCH_SIZE" == "" ]; then
		echo "ERROR: inceptionv1 model need to set batch_size"
		exit 1
	else
		remainder=$((BATCH_SIZE % (NODES*CORES)))
		if [ $remainder -ne 0 ]; then
			echo "ERROR: batch_size must be a multiple of 'nodes*cores'"
			exit 1
		fi
	fi
	DATA_DIR=$DATA_DIR_IMAGENET
	if [ -d $DATA_DIR/train ] && [ -d $DATA_DIR/val ]; then
		echo "Using existing data dir $DATA_DIR"
		HDFS_HOST=`echo $HDFS_DATA_DIR | awk -F '/' '{print $3}' | awk -F':' '{print $1}'`
		HADOOP_HOME=$HOME/hadoop-2.7.3
		if [ ! -d "hadoop-2.7.3" ]; then
			wget http://apache.claz.org/hadoop/common/hadoop-2.7.3/hadoop-2.7.3.tar.gz
			tar -xzf hadoop-2.7.3.tar.gz

cat << EOF > $HADOOP_HOME/etc/hadoop/core-site.xml
<configuration>
        <property>
                <name>fs.defaultFS</name>
                <value>hdfs://$HDFS_HOST:9000</value>
        </property>
</configuration>
EOF

		fi

		$HADOOP_HOME/bin/hadoop fs -mkdir -p $HDFS_DATA_DIR
		$HADOOP_HOME/bin/hadoop fs -copyFromLocal $DATA_DIR/* $HDFS_DATA_DIR
	else
		echo "Using HDFS data input: $HDFS_DATA_DIR"
	fi
fi

if [ -d $MODEL_DIR ]; then
	echo "Using existing model output dir $MODEL_DIR"
else
	echo "Making new dir for model output $MODEL_DIR"
	mkdir $MODEL_DIR
fi


[[ ! $CLASS =~ train|test ]] && {
	CLASS=train
}

cd $CURRENT

if [ "$MODEL" == "lenet" ] || [ "$MODEL" == "vgg" ]; then
	if [ "$CLASS" == "train" ]; then	     
		$SPARK_SUBMIT \
			$SPARK_URL \
			--total-executor-cores $(($CORES * $NODES)) \
			--executor-cores $CORES \
			--driver-cores $CORES \
			--driver-memory $MEMORY \
			--executor-memory $MEMORY \
			--num-executors $NODES \
			--class com.intel.analytics.bigdl.models.$MODEL.Train $BIGDL_JAR -f $DATA_DIR/ -b $BATCH_SIZE --maxEpoch $MAX_EPOCH --overWrite --checkpoint $MODEL_DIR
	else
		$SPARK_SUBMIT \
			$SPARK_URL \
            --total-executor-cores $(($CORES * $NODES)) \
            --executor-cores $CORES \
			--driver-cores $CORES \
			--driver-memory $MEMORY \
			--executor-memory $MEMORY \
			--num-executors $NODES \
			--class com.intel.analytics.bigdl.models.$MODEL.Test $BIGDL_JAR -f $DATA_DIR/ --model $TRAINED_MODEL -b $BATCH_SIZE
	fi
elif [ "$MODEL" == "inception-v1" ]; then
#echo $SPARK_URL
#echo $BIGDL_JAR
	if [ "$CLASS" == "train" ]; then
		$SPARK_SUBMIT \
			$SPARK_URL \
            --total-executor-cores $(($CORES * $NODES))  \
            --executor-cores $CORES  \
			--driver-cores $CORES \
			--driver-memory $MEMORY \
			--executor-memory $MEMORY \
			--num-executors $NODES \
			--driver-class-path $BIGDL_JAR \
			--class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 $BIGDL_JAR --batchSize $BATCH_SIZE --maxEpoch $MAX_EPOCH --overWrite --learningRate $LEARNING_RATE -f $HDFS_DATA_DIR --checkpoint $MODEL_DIR
	else
		$SPARK_SUBMIT \
			$SPARK_URL \
			--driver-cores $CORES \
			--driver-memory $MEMORY \
            --total-executor-cores $(($CORES * $NODES))  \
			--executor-cores $CORES  \
			--executor-memory $MEMORY \
			--num-executors $NODES \
			--driver-class-path $BIGDL_JAR \
			--class com.intel.analytics.bigdl.models.inception.Test $BIGDL_JAR --batchSize $BATCH_SIZE -f $HDFS_DATA_DIR/val
	fi
elif [ "$MODEL" == "perf" ]; then
	$SPARK_SUBMIT \
		$SPARK_URL \
		--driver-cores $CORES \
		--driver-memory $MEMORY \
        --total-executor-cores $(($CORES * $NODES))  \
		--executor-cores $CORES  \
		--executor-memory $MEMORY \
		--num-executors $NODES \
		--class com.intel.analytics.bigdl.models.utils.DistriOptimizerPerf $BIGDL_JAR -m inception_v1 --maxEpoch $MAX_EPOCH
fi


