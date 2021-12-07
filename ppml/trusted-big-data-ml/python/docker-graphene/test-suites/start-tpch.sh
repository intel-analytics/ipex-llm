#!/bin/bash
BASE_DIR=/ppml/trusted-big-data-ml
WORK_DIR=$BASE_DIR/work
TPCH_SPARK_DIR=$WORK_DIR/zoo-tutorials/tpch-spark
LOCAL_IP=192.168.0.112

cd $WORK_DIR

export http_proxy=$http_proxy
export https_proxy=$https_proxy
export SBT_OPTS="-Xms512M -Xmx1536M -Xss1M -XX:+CMSClassUnloadingEnabled -XX:MaxPermSize=256M"

# clone tpch
git clone https://github.com/intel-analytics/zoo-tutorials.git
cd $TPCH_SPARK_DIR
sed -i 's/2.11.7/2.12.1/g' tpch.sbt
sed -i 's/2.4.0/3.1.2/g' tpch.sbt

wget -P /usr/local https://github.com/sbt/sbt/releases/download/v1.5.5/sbt-1.5.5.tgz
tar -xvf /usr/local/sbt-1.5.5.tgz -C /usr/local/

# make & dbgen
cd  $TPCH_SPARK_DIR/dbgen
make
./dbgen -s $1

# sbt package
cd $TPCH_SPARK_DIR
java $SBT_OPTS -jar /usr/local/sbt/bin/sbt-launch.jar package

# run tpch
cd $BASE_DIR
SGX=1 ./pal_loader bash -c "export PYSPARK_PYTHON=/usr/bin/python && \
        ${JAVA_HOME}/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/zoo-tutorials/tpch-spark/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar:/ppml/trusted-big-data-ml/work/zoo-tutorials/tpch-spark/dbgen/*:/ppml/trusted-big-data-ml/work/spark-3.1.2/conf/:/ppml/trusted-big-data-ml/work/spark-3.1.2/jars/*' \
        -Xmx10g \
        -Dbigdl.mklNumThreads=1 \
        -XX:ActiveProcessorCount=24 \
        org.apache.spark.deploy.SparkSubmit \
        --master 'local[4]' \
        --conf spark.driver.port=10027 \
        --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
        --conf spark.worker.timeout=600 \
        --conf spark.starvation.timeout=250000 \
        --conf spark.rpc.askTimeout=600 \
        --conf spark.blockManager.port=10025 \
        --conf spark.driver.host=$LOCAL_IP \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --conf spark.sql.shuffle.partitions=8 \
        --class main.scala.TpchQuery \
        --conf spark.python.use.daemon=false \
        --conf spark.python.worker.reuse=false \
        --driver-memory 10G \
        /ppml/trusted-big-data-ml/work/zoo-tutorials/tpch-spark/target/scala-2.12/spark-tpc-h-queries_2.12-1.0.jar \
        /ppml/trusted-big-data-ml/work/zoo-tutorials/tpch-spark/dbgen \
        hdfs://$LOCAL_IP:9000/dbgen-query" 2>&1 | tee spark.local.tpc.h.sgx.log
