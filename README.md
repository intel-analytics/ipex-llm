#BigDL

A scalable deep learning library for Apache Spark.

Here's the summary of core features:
* a powerful N-dimensional array
* lots of math and data manipulating operations
* rich neural network layers
* effecient distributed numeric optimization routines on Apache Spark
* powered by MKL and MKL DNN, fast and optmized on Intel hardware platforms

##How to build
###Linux (only intel64 architecture)
####Build
<code>mvn clean package -DskipTests</code><br> Module native is skipped by default. If you want to build native module, see the [Full Build](#full-build).

####Full build
1. Download [Intel MKL](https://software.intel.com/en-us/intel-mkl) and install it in your linux box
2. Prepare MKL build environment<br>  <code>source PATH_TO_MKL/bin/mklvars.sh intel64</code><br> If Intel MKL doesn't install to default path /opt/intel, please link your libiomp5.so to project.<br> <code>ln -sf PATH_TO_INTEL_HOME/lib/intel64/libiomp5.so BIGDL_HOME/native/jni/src/main/resources/intel64</code>
3. Full build <br>  <code>mvn clean package -DskipTests -P full-build</code>

##Example
* MNIST example
* Cifar10 example
* Imagenet example
