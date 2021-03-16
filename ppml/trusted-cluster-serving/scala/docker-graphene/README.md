# trusted-cluster-serving
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
    -t intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT -f ./Dockerfile .
```

## How To Run
### Prepare the keys
The ppml in analytics zoo need secured keys to enable flink TLS, https and tlse enabled Redis, you need to prepare the secure keys and keystores.
```bash
    mkdir keys && cd keys
    openssl genrsa -des3 -out server.key 2048 ```
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
You also need to store the password you used in previous step in a secured file:
```
    mkdir password && cd password
    openssl genrsa -out key.txt 2048
    echo "YOUR_PASSWORD" | openssl rsautl -inkey key.txt -encrypt >output.bin
```

### Run the PPML Docker image
#### In local mode
##### Start the container to run analytics zoo cluster serving in ppml.
```bash
export KEYS_PATH=the_dir_path_of_your_prepared_keys
export SECURE_PASSWORD_PATH=the_dir_path_of_your_prepared_password
export LOCAL_IP=your_local_ip_of_the_sgx_server
sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-30" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $KEYS_PATH:/ppml/trusted-cluster-serving/redis/work/keys \
    -v $KEYS_PATH:/ppml/trusted-cluster-serving/java/work/keys \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-cluster-serving/redis/work/passowrd \
    -v $SECURE_PASSWORD_PATH:/ppml/trusted-cluster-serving/java/work/passowrd \
    --name=flink-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e CORE_NUM=30 \
    intelanalytics/analytics-zoo-ppml-trusted-cluster-serving-scala-graphene:0.10-SNAPSHOT /ppml/trusted-cluster-serving/start-all.sh
```

