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
    -t analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:0.10-SNAPSHOT -f ./Dockerfile .
```

## How to Run

### Prepare the data
To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example.
You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist).
There're four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page.
After you uncompress the gzip files, these files may be renamed by some uncompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example. 

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

### Run the PPML Docker image

#### In spark local mode
##### Start the container to run analytics zoo model training in ppml.
```bash
export DATA_PATH=the_dir_path_of_your_prepared_data
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export LOCAL_IP=your_local_ip_of_the_sgx_server
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
    intelanalytics/analytics-zoo-ppml-trusted-big-data-ml-scala-graphene:latest \
    bash -c "cd /ppml/trusted-big-data-ml/ && ./init.sh && ./start-spark-local-train-sgx.sh"
```
check the log:
```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/spark.local.sgx.log | egrep "###|INFO"
```
or
```bash
sudo docker logs spark-local | egrep "###|INFO"
```

#### In spark standalone cluster mode
