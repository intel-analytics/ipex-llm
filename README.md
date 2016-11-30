#BigDL

A scalable deep learning library for Apache Spark.

Here's the summary of core features:
* a powerful N-dimensional array
* lots of math and data manipulating operations
* rich neural network layers
* effecient distributed numeric optimization routines on Apache Spark
* powered by MKL and MKL DNN, fast and optmized on Intel hardware platforms

##How to build
###Linux
1. Download [Intel MKL](https://software.intel.com/en-us/intel-mkl) and install it in your linux box
2. Prepare MKL build environment<br>  <code>source PATH_TO_MKL/bin/mklvars.sh &#60;arch&#62;</code><br>  The **&#60;arch&#62;** can be *ia32*, *intel64*, or *mic*, which depends on your system.<br>Link your libiomp5.so to project.<br> <code>ln -sf PATH_TO_INTEL_HOME/lib/intel64/libiomp5.so native/jni/src/main/resources/intel64</code>
3. Build project<br>  <code>mvn clean package -DskipTests -P mkl</code>

##Example
* MNIST example
* Cifar10 example
* Imagenet example
