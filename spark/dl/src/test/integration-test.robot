*** Settings ***
Documentation    BigDL Integration Test
Resource         common.robot
Suite Setup      Prepare DataSource And Verticals
Suite Teardown   Delete All Sessions
Test template    BigDL Test

*** Test Cases ***   SuiteName
1                    Spark2.2 Test Suite
2                    Hdfs Test Suite
3                    Spark1.6 on Yarn Test Suite
4                    Spark2.3 on Yarn Test Suite
5                    Quantization Test Suite
6                    PySpark2.2 Test Suite
7                    PySpark3.0 Test Suite
8                    Spark3.0 on Yarn Test Suite

*** Keywords ***
Build SparkJar
   [Arguments]       ${spark_version}
   ${build}=         Catenate                        SEPARATOR=/    ${curdir}    make-dist.sh
   Log To Console    ${spark_version}
   Log To Console    start to build jar ${build} -P ${spark_version}
   Run               ${build} -P ${spark_version}
   Remove File       ${jar_path}
   Copy File         spark/dl/target/bigdl-${version}-jar-with-dependencies.jar    ${jar_path}
   Log To Console    build jar finished

DownLoad Input
   ${hadoop}=                       Catenate       SEPARATOR=/    /opt/work/hadoop-2.7.2/bin    hadoop
   Run                              ${hadoop} fs -get ${mnist_data_source} /tmp/mnist
   Log To Console                   got mnist data!! ${hadoop} fs -get ${mnist_data_source} /tmp/mnist
   Run                              ${hadoop} fs -get ${cifar_data_source} /tmp/cifar
   Log To Console                   got cifar data!! ${hadoop} fs -get ${cifar_data_source} /tmp/cifar
   Run                              ${hadoop} fs -get ${public_hdfs_master}:9000/text_data /tmp/
   Run                              tar -zxvf /tmp/text_data/20news-18828.tar.gz -C /tmp/text_data
   Log To Console                   got textclassifier data
   Set Environment Variable         http_proxy                                                  ${http_proxy}
   Set Environment Variable         https_proxy                                                 ${https_proxy}
   Run                              wget ${tiny_shakespeare}
   Set Environment Variable         LANG                                                        en_US.UTF-8
   Run                              head -n 8000 input.txt > val.txt
   Run                              tail -n +8000 input.txt > train.txt
   Run                              wget ${simple_example}
   Run                              tar -zxvf simple-examples.tgz
   Log To Console                   got examples data!!
   Create Directory                 model
   Create Directory                 models
   Remove Environment Variable      http_proxy                  https_proxy                     LANG

Remove Input
   Remove Directory                 model                       recursive=True
   Remove Directory                 models                      recursive=True
   Remove Directory                 /tmp/mnist                  recursive=True
   Remove File                      input.txt
   Remove Directory                 simple-examples             recursive=True
   Remove File                      simple-examples.tgz
   Remove Directory                 /tmp/text-data              recursive=True

Run Spark Test 
   [Arguments]                      ${submit}                   ${spark_master}
   DownLoad Input
   Log To Console                   begin lenet Train ${submit} --master ${spark_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 5g --executor-cores 16 --total-executor-cores 32 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 256 -e 3 --optimizerVersion "optimizerV2"
   Run Shell                        ${submit} --master ${spark_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 5g --executor-cores 16 --total-executor-cores 32 --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 256 -e 3 --optimizerVersion "optimizerV2"
   Log To Console                   begin lenet Train local[4]
   Run Shell                        ${submit} --master local[4] --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f /tmp/mnist -b 120 -e 1 --optimizerVersion "optimizerV2"
   Log To Console                   begin autoencoder Train 
   Run Shell                        ${submit} --master ${spark_master} --executor-cores 4 --total-executor-cores 8 --class com.intel.analytics.bigdl.models.autoencoder.Train ${jar_path} -b 120 -e 1 -f /tmp/mnist --optimizerVersion "optimizerV2"
   Log To Console                   begin PTBWordLM
   Run Shell                        ${submit} --master ${spark_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 40g --executor-memory 40g --executor-cores 8 --total-executor-cores 8 --class com.intel.analytics.bigdl.example.languagemodel.PTBWordLM ${jar_path} -f ./simple-examples/data -b 120 --numLayers 2 --vocab 10001 --hidden 650 --numSteps 35 --learningRate 0.005 -e 1 --learningRateDecay 0.001 --keepProb 0.5 --overWrite --optimizerVersion "optimizerV2"
   Log To Console                   begin resnet Train
   Run Shell                        ${submit} --master ${spark_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 5g --executor-memory 5g --executor-cores 8 --total-executor-cores 32 --class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 ${jar_path} -f /tmp/cifar --batchSize 448 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 1 --learningRate 0.1 --optimizerVersion "optimizerV2"
   Log To Console                   begin DLClassifierLeNet
   Run Shell                        ${submit} --master ${spark_master} --executor-cores 16 --total-executor-cores 16 --driver-memory 5g --executor-memory 30g --class com.intel.analytics.bigdl.example.MLPipeline.DLClassifierLeNet ${jar_path} -b 1200 -f /tmp/mnist --maxEpoch 1 
   Log To Console                   begin rnn Train
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --executor-cores 12 --total-executor-cores 12 --class com.intel.analytics.bigdl.models.rnn.Train ${jar_path} -f ./ -s ./models --nEpochs 1 --checkpoint ./model/ -b 12 --optimizerVersion "optimizerV2"
   Log To Console                   begin inceptionV1 train
   Run Shell                        ${submit} --master ${spark_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 20g --executor-memory 40g --executor-cores 10 --total-executor-cores 20 --class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 ${jar_path} -b 40 -f ${imagenet_test_data_source} --learningRate 0.1 -i 100 --optimizerVersion "optimizerV2"
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master ${spark_master} --driver-memory 5g --executor-memory 5g --total-executor-cores 32 --executor-cores 8 --class com.intel.analytics.bigdl.example.textclassification.TextClassifier ${jar_path} --batchSize 128 --baseDir /tmp/text_data --partitionNum 32 
   Remove Input

Spark2.2 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME               /opt/work/spark-2.2.0-bin-hadoop2.7
   ${submit}=                       Catenate                 SEPARATOR=/    /opt/work/spark-2.2.0-bin-hadoop2.7/bin    spark-submit
   Run Spark Test                   ${submit}                ${spark_22_master} 

Hdfs Test Suite
   Set Environment Variable         hdfsMaster               ${hdfs_272_master}
   Set Environment Variable         mnist                    ${mnist_data_source}
   Set Environment Variable         s3aPath                  ${s3a_path}
   Run Shell                        mvn clean test -Dsuites=com.intel.analytics.bigdl.integration.HdfsSpec -DhdfsMaster=${hdfs_272_master} -Dmnist=${mnist_data_source} -P integration-test -DforkMode=never
   Run Shell                        mvn clean test -Dsuites=com.intel.analytics.bigdl.integration.S3Spec -Ds3aPath=${s3a_path} -P integration-test -DforkMode=never
   Run Shell                        mvn clean test -Dsuites=com.intel.analytics.bigdl.optim.OptimPredictorShutdownSpec  -DhdfsMaster=${hdfs_272_master} -P integration-test -DforkMode=never
   Remove Environment Variable      hdfsMaster               mnist                   s3aPath


Quantization Test Suite
   ${hadoop}=                       Catenate                 SEPARATOR=/             /opt/work/hadoop-2.7.2/bin        hadoop
   Run                              ${hadoop} fs -get ${mnist_data_source} /tmp/
   Log To Console                   got mnist data!!
   Run                              ${hadoop} fs -get ${cifar_data_source} /tmp/
   Log To Console                   got cifar data!!
   Set Environment Variable         mnist                    /tmp/mnist
   Set Environment Variable         cifar10                  /tmp/cifar
   Set Environment Variable         lenetfp32model           ${public_hdfs_master}:9000/lenet4IT4J1.7B4.bigdl
   Set Environment Variable         resnetfp32model          ${public_hdfs_master}:9000/resnet4IT4J1.7B4.bigdl
   Remove Environment Variable      mnist                    cifar10                 lenetfp32model                  resnetfp32model

Spark1.6 on Yarn Test Suite
   Yarn Test Suite	spark_1.6	/opt/work/spark-1.6.0-bin-hadoop2.6
   
Spark2.3 on Yarn Test Suite
   Yarn Test Suite	spark_2.x	/opt/work/spark-2.3.1-bin-hadoop2.7

Spark3.0 on Yarn Test Suite
   Yarn Test Suite      spark_3.x       /opt/work/spark-3.0.0-bin-hadoop2.7

Yarn Test Suite
   [Arguments]                      ${bigdl_spark_version}                ${spark_home}
   DownLoad Input
   Build SparkJar                   ${bigdl_spark_version}
   Set Environment Variable         SPARK_HOME               ${spark_home}  
   Set Environment Variable         http_proxy               ${http_proxy}
   Set Environment Variable         https_proxy              ${https_proxy}
   ${submit}=                       Catenate                 SEPARATOR=/    ${spark_home}	bin    spark-submit
   Log To Console                   begin DLClassifierLeNet
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 1 --driver-memory 20g --executor-memory 60g --class com.intel.analytics.bigdl.example.MLPipeline.DLClassifierLeNet ${jar_path} -b 1200 -f /tmp/mnist --maxEpoch 1
   Log To Console                   begin text classification
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --conf spark.yarn.executor.memoryOverhead=40000 --executor-cores 10 --num-executors 2 --driver-memory 20g --executor-memory 40g --class com.intel.analytics.bigdl.example.textclassification.TextClassifier ${jar_path} --batchSize 240 --baseDir /tmp/text_data --partitionNum 4
   Log To Console                   begin lenet
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 3 --driver-memory 20g --class com.intel.analytics.bigdl.models.lenet.Train ${jar_path} -f ${mnist_data_source} -b 120 -e 3 --optimizerVersion "optimizerV2"
   Log To Console                   begin autoencoder Train 
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 3 --driver-memory 20g --class com.intel.analytics.bigdl.models.autoencoder.Train ${jar_path} -b 120 -e 1 -f /tmp/mnist --optimizerVersion "optimizerV2"
   Log To Console                   begin resnet Train
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 3 --driver-memory 20g --class com.intel.analytics.bigdl.models.resnet.TrainCIFAR10 ${jar_path} -f /tmp/cifar --batchSize 120 --optnet true --depth 20 --classes 10 --shortcutType A --nEpochs 1 --learningRate 0.1 --optimizerVersion "optimizerV2"
   Log To Console                   begin rnn Train
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 3 --driver-memory 20g --class com.intel.analytics.bigdl.models.rnn.Train ${jar_path} -f ./ -s ./models --nEpochs 1 --checkpoint ./model/ -b 120 --optimizerVersion "optimizerV2"
   Log To Console                   begin PTBWordLM
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 8 --num-executors 1 --driver-memory 20g --executor-memory 40g --class com.intel.analytics.bigdl.example.languagemodel.PTBWordLM ${jar_path} -f ./simple-examples/data -b 120 --numLayers 2 --vocab 10001 --hidden 650 --numSteps 35 --learningRate 0.005 -e 1 --learningRateDecay 0.001 --keepProb 0.5 --overWrite --optimizerVersion "optimizerV2"
   Log To Console                   begin inceptionV1 train
   Run Shell                        ${submit} --master yarn --deploy-mode client --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --executor-cores 10 --num-executors 2 --driver-memory 20g --executor-memory 40g --class com.intel.analytics.bigdl.models.inception.TrainInceptionV1 ${jar_path} -b 40 -f ${imagenet_test_data_source} --learningRate 0.1 -i 100 --optimizerVersion "optimizerV2"
   Run Shell                        ${submit} --master yarn --deploy-mode client --executor-memory 2g --driver-memory 2g --executor-cores 10 --num-executors 2 --properties-file ${curdir}/dist/conf/spark-bigdl.conf --jars ${jar_path} --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --conf spark.driver.extraClassPath=${jar_path} --conf spark.executor.extraClassPath=bigdl-${version}-jar-with-dependencies.jar ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 200 --action train --endTriggerType epoch --endTriggerNum 1 
   Remove Environment Variable      http_proxy                https_proxy
   Remove Input
   

PySpark2.2 Test Suite
   Build SparkJar                   spark_2.x
   Set Environment Variable         SPARK_HOME     /opt/work/spark-2.2.0-bin-hadoop2.7
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-2.2.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_22_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 10g --executor-cores 14 --total-executor-cores 28 --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --jars ${jar_path} --properties-file ${curdir}/dist/conf/spark-bigdl.conf ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 224 --action train --endTriggerType epoch --endTriggerNum 1

PySpark3.0 Test Suite
   Build SparkJar                   spark_3.x
   Set Environment Variable         SPARK_HOME     /opt/work/spark-3.0.0-bin-hadoop2.7
   ${submit}=                       Catenate       SEPARATOR=/    /opt/work/spark-3.0.0-bin-hadoop2.7/bin    spark-submit
   Run Shell                        ${submit} --master ${spark_30_master} --conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" --driver-memory 10g --executor-cores 14 --total-executor-cores 28 --py-files ${curdir}/dist/lib/bigdl-${version}-python-api.zip --jars ${jar_path} --properties-file ${curdir}/dist/conf/spark-bigdl.conf ${curdir}/pyspark/bigdl/models/lenet/lenet5.py -b 224 --action train --endTriggerType epoch --endTriggerNum 1
