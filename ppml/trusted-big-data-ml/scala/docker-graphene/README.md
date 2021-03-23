# trusted-big-data-ml
Please mind the ip and file path settings, they should be changed to the ip/path of your own sgx server on which you are running the programs.


## How To Build 

```bash
export HTTP_PROXY_HOST=your_http_proxy_host
export HTTP_PROXY_PORT=your_http_proxy_port
export HTTPS_PROXY_HOST=your_https_proxy_host
export HTTPS_PROXY_PORT=your_https_proxy_port
export JDK_URL=http://your-http-url-to-download-jdk
sudo docker build \
    --build-arg http_proxy=http://$HTTP_PROXY_HOST:$HTTP_PROXY_PORT \
    --build-arg https_proxy=http://$HTTPS_PROXY_HOST:$HTTPS_PROXY_PORT \
    --build-arg HTTP_PROXY_HOST=$HTTP_PROXY_HOST \
    --build-arg HTTP_PROXY_PORT=$HTTP_PROXY_PORT \
    --build-arg HTTPS_PROXY_HOST=$HTTPS_PROXY_HOST \
    --build-arg HTTPS_PROXY_PORT=$HTTPS_PROXY_PORT \
    --build-arg JDK_VERSION=8u192 \
    --build-arg JDK_URL=$JDK_URL \
    --build-arg no_proxy=x.x.x.x \
    -t intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT -f ./Dockerfile .
```

## How to Run

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example. <br>
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). <br>
There're four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. <br>
After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.  <br>

### Prepare the keys
The ppml in analytics zoo need secured keys to enable spark security such as AUTHENTICATION, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores.
```bash
    mkdir keys && cd keys
    openssl genrsa -des3 -out server.key 2048
    openssl req -new -key server.key -out server.csr
    openssl x509 -req -days 9999 -in server.csr -signkey server.key -out server.crt
    cat server.key > server.pem
    cat server.crt >> server.pem
    openssl pkcs12 -export -in server.pem -out keystore.pkcs12
    keytool -importkeystore -srckeystore keystore.pkcs12 -destkeystore keystore.jks -srcstoretype PKCS12 -deststoretype JKS
    openssl pkcs12 -in keystore.pkcs12 -nodes -out server.pem
    openssl rsa -in server.pem -out server.key
    openssl x509 -in server.pem -out server.crt
```

### Run PPML Docker image

#### In spark local mode
##### Start container to run tests in ppml
```bash
export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server

sudo docker pull 10.239.45.10/arda/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT
    
sudo docker exec -it spark-local bash
cd ppml/trusted-bid-data-ml
```

##### Example Test 1 
```bash
./init.sh
vim start-spark-local-pi-sgx.sh
```
Add these code in the `start-spark-local-pi-sgx.sh` file: <br>
```bash
#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
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
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class org.apache.spark.examples.SparkPi \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/spark-2.4.3/examples/jars/spark-examples_2.11-2.4.3.jar | tee spark.local.pi.sgx.log
```

Then run the script to run pi test in spark: <br>
```bash
./start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.pi.sgx.log | egrep "###|INFO|Pi"
```

The result should look like: <br>
>   Pi is roughly 3.1422957114785572

##### Example Test 2
```bash
./init.sh
./start-spark-local-train-sgx.sh
```

Open another terminal and check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.sgx.log | egrep "###|INFO"
```
or
```bash
sudo docker logs spark-local | egrep "###|INFO"
```

The result should look like: <br>
>   ############# train optimized[P1182:T2:java] ---- end time: 310534 ms return from shim_write(...) = 0x1d <br>
>   ############# ModuleLoader.saveToFile File.saveBytes end, used 827002 ms[P1182:T2:java] ---- end time: 1142754 ms return from shim_write(...) = 0x48 <br>
>   ############# ModuleLoader.saveToFile saveWeightsToFile end, used 842543 ms[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x4b <br>
>   ############# model saved[P1182:T2:java] ---- end time: 1985297 ms return from shim_write(...) = 0x19 <br>

##### Example Test 3
Before run TPC-H test in container we created, we should download and install [SBT](https://www.scala-sbt.org/download.html), then build and package TPC-H dataset according to [TPC-H](https://github.com/qiuxin2012/tpch-spark) with your needs. After packaged, check if we have `spark-tpc-h-queries_2.11-1.0.jar ` under `/tpch-spark/target/scala-2.11`, if have, we package successfully.

Copy TPC-H to container: <br>
```bash
docker cp tpch-spark/ spark-local:/ppml/trusted-big-data-ml/work
sudo docker exec -it spark-local bash
cd ppml/trusted-big-data-ml/
./init.sh
vim start-spark-local-tpc-h-sgx.sh
```

Add these code in the `start-spark-local-tpc-h-sgx.sh` file: <br>
```bash
#!/bin/bash

set -x

SGX=1 ./pal_loader /opt/jdk8/bin/java \
        -cp '/ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar:/ppml/trusted-big-data-ml/work/tpch-spark/dbgen/*:/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
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
        --conf spark.driver.host=127.0.0.1 \
        --conf spark.driver.blockManager.port=10026 \
        --conf spark.io.compression.codec=lz4 \
        --class main.scala.TpchQuery \
        --executor-cores 4 \
        --total-executor-cores 4 \
        --executor-memory 10G \
        /ppml/trusted-big-data-ml/work/tpch-spark/target/scala-2.11/spark-tpc-h-queries_2.11-1.0.jar \
        /ppml/trusted-big-data-ml/work/tpch-spark/dbgen | tee spark.local.tpc.h.sgx.log
```

Then run the script to run TPC-H test in spark: <br>
```bash
./start-spark-local-tpc-h-sgx.sh
```

Open another terminal and check the log: <br>
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.tpc.h.sgx.log | egrep "###|INFO"
```

#### In spark standalone cluster mode
##### setup passwordless ssh login to all the nodes.
##### config the environments for master, workers, docker image, security keys/passowrd files and data path.
```bash
nano environments.sh
```
##### start the distributed cluster serving
```bash
./start-distributed-big-data-ml.sh
```
##### stop the distributed cluster serving 
```bash
./stop-distributed-big-data-ml.sh
```
