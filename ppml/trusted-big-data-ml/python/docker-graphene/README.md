# Trusted Big Data ML with Python

SGX-based Trusted Big Data ML allows the user to run end-to-end big data analytics application and Intel Analytics Zoo and BigDL model training with spark local and distributed cluster on Graphene-SGX.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running the programs.*

## Before Running the Code

#### 1. Build docker image

Before running the following command, please modify the paths in `build-docker-image.sh`. Then build the docker image with the following command.

```bash
./build-docker-image.sh
```

#### 2. Prepare data, key and password

*  ##### Prepare the Data

  To train a model with ppml in analytics zoo and bigdl, you need to prepare the data first. The Docker image is taking lenet and mnist as example.You can download the MNIST Data from [here](http://yann.lecun.com/exdb/mnist/). Unzip all the files and put them in one folder(e.g. mnist). 

  There are four files. **train-images-idx3-ubyte** contains train images, **train-labels-idx1-ubyte** is train label file, **t10k-images-idx3-ubyte** has validation images and **t10k-labels-idx1-ubyte** contains validation labels. For more detail, please refer to the download page. 

  After you decompress the gzip files, these files may be renamed by some decompress tools, e.g. **train-images-idx3-ubyte** is renamed to **train-images.idx3-ubyte**. Please change the name back before you run the example.

* ##### Prepare the Key

  The ppml in analytics zoo needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

  ```bash
  sudo ../../../scripts/generate-keys.sh
  ```

  You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

  It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

  ```bash
  openssl genrsa -3 -out enclave-key.pem 3072
  ```

* ##### Prepare the Password

  Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

  ```bash
  sudo bash ../../../scripts/generate-password.sh used_password_when_generate_keys
  ```



## Run as Spark Local Mode

#### 1. Start the container to run spark applications in spark local mode

Before you run the following commands to start the container, you need to modify the paths in `deploy-local-spark-sgx.sh` and then run the following commands.

```bash
./deploy-local-spark-sgx.sh
sudo docker exec -it spark-local bash
cd /ppml/trusted-big-data-ml
./init.sh
```

 #### 2. Run native python examples

##### Example 1: `helloworld.py`

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \ 
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/helloworld.py" | tee test-helloworld-sgx.log
```
Then check the output with the following command.

```bash
cat test-helloworld-sgx.log | egrep "Hello World"
```

The result should be 

> Hello World



##### Example 2: `test-numpy.py`

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \ 
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/test-numpy.py" | tee test-numpy-sgx.log
```

Then check the output with the following command.

```bash
cat test-numpy-sgx.log | egrep "numpy.dot"
```

The result should be similar to

> numpy.dot: 0.034211914986371994 sec



##### Example3: `pytorch`

Before running the pytorch example, first you need to download the pretrained model.
```bash
cd work/examples/pytorch/
python download-pretrained-model.py
cd ../..
```

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/pytorch/pytorchexample.py" | tee test-pytorch-sgx.log
```

Then check the output with the following command.

```bash
cat result.txt
```

The result should be

>[('Labrador retriever', 41.585174560546875), ('golden retriever', 16.591665267944336), ('Saluki, gazelle hound', 16.286867141723633), ('whippet', 2.8539085388183594), ('Ibizan hound, Ibizan Podenco', 2.392474889755249)]



##### Example4: `tensorflow-lite`

Before running the tensorflow example, first you need to download the pretrained model and other dependant files.

```bash
cd work/examples/tensorflow-lite/
make install-dependencies-ubuntu
cd ../..
```

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow-lite/label_image \
	-m /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/inception_v3.tflite \
	-i /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/image.bmp \
	-l /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/labels.txt" | tee test-tflite-sgx.log
```

Then check the output with the following command.

```bash
cat test-tflite-sgx.log | egrep "military"
```

The result should be

>0.699143: 653 military uniform



##### Example5: `tensorflow`

Run the example with SGX with the following command in the terminal.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow/hand_classifier_with_resnet.py" | tee test-tf-sgx.log
```

Then check the output with the following command.

```bash
cat test-tf-sgx.log | egrep "accuracy"
```

The result should be similar to

>test accuracy 0.78357
>
>training data accuracy 0.92361



## Run as Spark Standalone Mode

#### 1. Start the container to run spark applications in spark standalone mode

Before you run the following commands to start the container, you need to modify the paths in `environment.sh` and then run the following commands.

```bash
./deploy-distributed-standalone-spark.sh
./start-distributed-spark-driver.sh
```

Then use `distributed-check-status.sh` to check master's and worker's status and make sure that both of them are running.

Use the following commands to enter the docker of spark driver.

```bash
sudo docker exec -it spark-driver bash
cd /ppml/trusted-big-data-ml
./start-spark-standalone-driver-sgx.sh
./init.sh
```

#### 2. Run pyspark examples

##### Example1: `pi.py`

Run the example with SGX and standalone mode with the following command in the terminal. Replace the value of ``spark.authenticate.secret`` with your own secret key.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'spark://192.168.0.111:7077' \
	--conf spark.authenticate=true \
  --conf spark.authenticate.secret=your_secret_key \
	/ppml/trusted-big-data-ml/work/spark-2.4.3/examples/src/main/python/pi.py" | tee test-pi-sgx.log

```

Then check the output with the following command.

```bash
cat test-pi-sgx.log | egrep "roughly"
```

The result should be similar to

>Pi is roughly 3.146760



##### Example2: `test-wordcount.py`

Run the example with SGX and standalone mode with the following command in the terminal. Replace the value of ``spark.authenticate.secret`` with your own secret key.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'spark://192.168.0.111:7077' \
	--conf spark.authenticate=true \
  --conf spark.authenticate.secret=your_secret_key \
	/ppml/trusted-big-data-ml/work/spark-2.4.3/examples/src/main/python/wordcount.py ./work/examples/helloworld.py" | tee test-wordcount-sgx.log

```

Then check the output with the following command.

```bash
cat test-wordcount-sgx.log | egrep "print"
```

The result should be similar to

> print("Hello: 1
>
> print(sys.path);: 1



##### Example3: Basic SQL

Run the example with SGX and standalone mode with the following command in the terminal. Replace the value of ``spark.authenticate.secret`` with your own secret key.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'spark://192.168.0.111:7077' \
	--conf spark.authenticate=true \
  --conf spark.authenticate.secret=your_secret_key \
	/ppml/trusted-big-data-ml/work/spark-2.4.3/examples/src/main/python/sql/basic.py" | tee test-sql-basic-sgx.log

```

Then check the output with the following command.

```bash
cat test-sql-basic-sgx.log | egrep "Justin"
```

The result should be similar to

> | 19|  Justin|
>
> |  Justin| 
>
> |  Justin|       20|
>
> | 19|  Justin|
>
> | 19|  Justin|
>
> | 19|  Justin|
>
> Name: Justin
>
> |  Justin|



##### Example4: Bigdl lenet

Run the example with SGX and standalone mode with the following command in the terminal. Replace the value of ``spark.authenticate.secret`` with your own secret key.

```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java -cp \
  '/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar:/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
  -Xmx8g \
  org.apache.spark.deploy.SparkSubmit \
  --master 'spark://192.168.0.111:7077' \
  --conf spark.authenticate=true \
  --conf spark.authenticate.secret=your_secret_key \
  --conf spark.driver.memory=8g \
  --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --conf spark.rpc.message.maxSize=190 \
  --conf spark.network.timeout=10000000 \
  --conf spark.executor.heartbeatInterval=10000000 \
  --py-files /ppml/trusted-big-data-ml/work/bigd-python-api.zip,/ppml/trusted-big-data-ml/work/examples/bigdl/lenet/lenet.py \
  --jars /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
  --driver-cores 2 \
  --total-executor-cores 2 \
  --executor-cores 2 \
  --executor-memory 8g \
  /ppml/trusted-big-data-ml/work/examples/bigdl/lenet/lenet.py \
  --dataPath /ppml/trusted-big-data-ml/work/data/mnist \
  --maxEpoch 2" | tee test-bigdl-lenet-sgx.log
```

Then check the output with the following command.

```bash
cat test-bigdl-lenet-sgx.log | egrep "Accuracy"
```

The result should be similar to

>creating: createTop1Accuracy
>
>2021-06-18 01:39:45 INFO DistriOptimizer$:180 - [Epoch 1 60032/60000][Iteration 469][Wall Clock 457.926565s] Top1Accuracy is Accuracy(correct: 9488, count: 10000, accuracy: 0.9488)
>
>2021-06-18 01:46:20 INFO DistriOptimizer$:180 - [Epoch 2 60032/60000][Iteration 938][Wall Clock 845.747782s] Top1Accuracy is Accuracy(correct: 9696, count: 10000, accuracy: 0.9696)

