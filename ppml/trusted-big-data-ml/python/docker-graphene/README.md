### Running Five Basic Examples with and without SGX

#### 1. Running `helloworld.py` Example
* To run the example with SGX, enter the following command in the terminal.
```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \ 
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/helloworld.py" | tee test-helloworld-sgx.log
```
* To run the example without SGX, enter the following commands in the terminal.
```bash
/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/helloworld.py | tee test-helloworld.log
```

#### 2. Running `test-numpy.py` Example
* To run the example with SGX, enter the following command in the terminal.
```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \ 
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/test-numpy.py" | tee test-numpy-sgx.log
```

* To run the example without SGX, enter the following commands in the terminal.
```bash
/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/test-numpy.py | tee test-numpy.log
```

#### 3. Running `pytorch` Example
Before running the pytorch example, first you need to download the pretrained model.
```bash
cd work/examples/pytorch/
python download-pretrained-model.py
cd ../..
```

* To run the example with SGX, enter the following command in the terminal.
```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/pytorch/pytorchexample.py" | tee test-pytorch-sgx.log
```

* To run the example without SGX, enter the following commands in the terminal.
```bash
/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/pytorch/pytorchexample.py | tee test-pytorch.log
```

#### 4. Running `tensorflow-lite` Example
Before running the tensorflow example, first you need to download the pretrained model and other dependant files.
```bash
cd work/examples/tensorflow-lite/
make install-dependencies-ubuntu
cd ../..
```

* To run the example with SGX, enter the following command in the terminal.
```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow-lite/label_image \
	-m /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/inception_v3.tflite \
	-i /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/image.bmp \
	-l /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/labels.txt" | tee test-pytorch-sgx.log
```

* To run the example without SGX, enter the following commands in the terminal.
```bash
/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow-lite/label_image \
	-m /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/inception_v3.tflite \
	-i /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/image.bmp \
	-l /ppml/trusted-big-data-ml/work/examples/tensorflow-lite/labels.txt | tee test-tflite.log
```

#### 5. Running `tensorflow` Example

* To run the example with SGX, enter the following command in the terminal.
```bash
SGX=1 ./pal_loader bash -c "/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow/hand_classifier_with_resnet.py" | tee test-tf-sgx.log
```

* To run the example without SGX, enter the following commands in the terminal.
```bash
/opt/jdk8/bin/java \
	-cp '/ppml/trusted-big-data-ml/work/spark-2.4.3/conf/:/ppml/trusted-big-data-ml/work/spark-2.4.3/jars/*' \
	-Xmx1g org.apache.spark.deploy.SparkSubmit \
	--master 'local[4]' \
	/ppml/trusted-big-data-ml/work/examples/tensorflow/hand_classifier_with_resnet.py | tee test-tf.log
```


