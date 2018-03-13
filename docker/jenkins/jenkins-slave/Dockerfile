FROM ubuntu:14.04

MAINTAINER The BigDL Authors https://github.com/intel-analytics/BigDL

WORKDIR /opt/work

ENV JAVA_7_HOME		/opt/work/jdk7
ENV JAVA_8_HOME		/opt/work/jdk8
ENV JAVA_HOME		${JAVA_7_HOME}
ENV CONDA_HOME		/opt/work/conda
ENV JENKINS_HOME	/opt/work/jenkins
ENV SPARK_1_5_HOME      /opt/work/spark-1.5.0
ENV SPARK_1_6_HOME      /opt/work/spark-1.6.0
ENV SPARK_2_1_HOME      /opt/work/spark-2.1.1
ENV SPARK_2_2_HOME      /opt/work/spark-2.2.0
ENV TORCH_VERSION       d03a42834bb1b674495b0c42de1716b66cc388f1
ENV TORCH_HOME		/opt/work/torch
ENV CAFFE_HOME      	/opt/work/caffe
ENV PATH                ${TORCH_HOME}/install/bin:${JAVA_HOME}/bin:${CONDA_HOME}/bin:${PATH}
ENV LANG 		en_US.UTF-8
ENV LC_ALL 		en_US.UTF-8

ADD ./run-slave.sh 	/opt/work/jenkins/run-slave.sh
ADD ./slave.groovy 	/opt/work/jenkins/slave.groovy
ADD ./pip.conf		/root/.pip/pip.conf
ADD ./settings.xml	/root/.m2/settings.xml

RUN apt-get update --fix-missing && \
    apt-get install -y vim curl nano wget unzip maven git && \
    apt-get install -y locales && locale-gen en_US.UTF-8

#jdk8
RUN wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 ${JAVA_8_HOME} && \
#jdk7
    wget http://ftp.osuosl.org/pub/funtoo/distfiles/oracle-java/jdk-7u80-linux-x64.tar.gz && \
    gunzip jdk-7u80-linux-x64.tar.gz && \
    tar -xf jdk-7u80-linux-x64.tar -C /opt && \
    rm jdk-7u80-linux-x64.tar && \
    ln -s /opt/jdk1.7.0_80 ${JAVA_7_HOME} && \
#Jenkins
    chmod a+x /opt/work/jenkins/run-slave.sh && \
    chmod a+x /opt/work/jenkins/slave.groovy && \
    wget http://repo.jenkins-ci.org/releases/org/jenkins-ci/main/remoting/3.14/remoting-3.14.jar && \
    mv remoting-3.14.jar ${JENKINS_HOME} && \
#spark 1.5.0 1.6.0 2.1.1 2.2.0
    wget http://archive.apache.org/dist/spark/spark-1.5.0/spark-1.5.0-bin-hadoop2.6.tgz && \
    tar -xf spark-1.5.0-bin-hadoop2.6.tgz -C /opt/work && \
    rm spark-1.5.0-bin-hadoop2.6.tgz && \
    ln -s /opt/work/spark-1.5.0-bin-hadoop2.6 ${SPARK_1_5_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-1.6.0/spark-1.6.0-bin-hadoop2.6.tgz && \
    tar -xf spark-1.6.0-bin-hadoop2.6.tgz -C /opt/work && \
    rm spark-1.6.0-bin-hadoop2.6.tgz && \
    ln -s /opt/work/spark-1.6.0-bin-hadoop2.6 ${SPARK_1_6_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-2.1.1/spark-2.1.1-bin-hadoop2.7.tgz && \
    tar -xf spark-2.1.1-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.1.1-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.1.1-bin-hadoop2.7 ${SPARK_2_1_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-2.2.0/spark-2.2.0-bin-hadoop2.7.tgz && \
    tar -xf spark-2.2.0-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.2.0-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.2.0-bin-hadoop2.7 ${SPARK_2_2_HOME}

#MKL
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler && \
    apt-get install -y --no-install-recommends libboost-all-dev && \
    apt-get install -y gcc g++ gfortran wget cpio && \
    wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/12414/l_mkl_2018.1.163.tgz && \
    tar -xzvf l_mkl_2018.1.163.tgz && \
    sed -i 's/ACCEPT_EULA=decline/ACCEPT_EULA=accept/g' /opt/work/l_mkl_2018.1.163/silent.cfg && \
    sed -i 's/ARCH_SELECTED=ALL/ARCH_SELECTED=INTEL64/g' /opt/work/l_mkl_2018.1.163/silent.cfg && \
    sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=;intel-comp-l-all-vars__noarch;intel-comp-nomcu-vars__noarch;intel-openmp__x86_64;intel-tbb-libs__x86_64;intel-mkl-common__noarch;intel-mkl-installer-license__noarch;intel-mkl-core__x86_64;intel-mkl-core-rt__x86_64;intel-mkl-doc__noarch;intel-mkl-doc-ps__noarch;intel-mkl-gnu__x86_64;intel-mkl-gnu-rt__x86_64;intel-mkl-common-ps__noarch;intel-mkl-core-ps__x86_64;intel-mkl-common-c__noarch;intel-mkl-core-c__x86_64;intel-mkl-common-c-ps__noarch;intel-mkl-tbb__x86_64;intel-mkl-tbb-rt__x86_64;intel-mkl-gnu-c__x86_64;intel-mkl-common-f__noarch;intel-mkl-core-f__x86_64;intel-mkl-gnu-f-rt__x86_64;intel-mkl-gnu-f__x86_64;intel-mkl-f95-common__noarch;intel-mkl-f__x86_64;intel-mkl-psxe__noarch;intel-psxe-common__noarch;intel-psxe-common-doc__noarch;intel-compxe-pset/g' /opt/work/l_mkl_2018.1.163/silent.cfg && \
    /opt/work/l_mkl_2018.1.163/install.sh -s /opt/work/l_mkl_2018.1.163/silent.cfg && \
    ln -sf /opt/intel/mkl/lib/intel64/libmkl_rt.so /usr/lib/libblas.so && \
    ln -sf /opt/intel/mkl/lib/intel64/libmkl_rt.so /usr/lib/libblas.so.3 && \
    ln -sf /opt/intel/mkl/lib/intel64/libmkl_rt.so /usr/lib/liblapack.so && \
    ln -sf /opt/intel/mkl/lib/intel64/libmkl_rt.so /usr/lib/liblapack.so.3 && \
    echo "/opt/intel/mkl/lib/intel64" > /etc/ld.so.conf.d/libblas.conf && \
    echo "/opt/intel/lib/intel64" >> /etc/ld.so.conf.d/libblas.conf && \
    ldconfig && \
    rm -rf /opt/work/l_mkl_2018.1.163.tgz /opt/work/l_mkl_2018.1.163    
    
ENV TENSOR_FLOW_VERION	1.2.0
ENV KERAS_VERSION	1.2.2

#python-native
RUN apt-get install -y --no-install-recommends build-essential tcl tk gettext autoconf zlib1g-dev gcc python-dev python3-dev libcurl4-openssl-dev && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python2 get-pip.py && \
    pip2 install --no-cache-dir robotframework robotframework-requests requests && \
    pip2 install numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytest pytest-xdist h5py ipython==5.0.0 tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    python3 get-pip.py && \
    pip install --no-cache-dir robotframework robotframework-requests requests && \
    pip install numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytest pytest-xdist h5py ipython==5.0.0 tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION

#python-conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.3.31-Linux-x86_64.sh -f -b -p ${CONDA_HOME} && \
    rm Miniconda3-4.3.31-Linux-x86_64.sh && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda update --all --quiet --yes && \
    conda clean -tipsy && \
    conda create -y -n py27 python=2.7 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytest pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda create -y -n py35 python=3.5 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytest pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda create -y -n py36 python=3.6 mkdocs numpy scipy pandas scikit-learn matplotlib seaborn jupyter plotly nltk twine pytest pytest-xdist h5py tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda info --env

#TENSORFLOW MODELS
ENV PYTHONPATH /opt/work/models/research/slim:${PYTHONPATH}
RUN git clone https://github.com/tensorflow/models.git

#Caffe
ENV PYTHONPATH=${CAFFE_HOME}/python:${CAFFE_HOME}/distribute/python:${PYTHONPATH}
RUN apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev && \
    git clone https://github.com/BVLC/caffe.git && \
    cd /opt/work/caffe && \
    cp /opt/work/caffe/Makefile.config.example Makefile.config && \
    echo "CPU_ONLY := 1" >> Makefile.config && \
    sed -i "s/BLAS := atlas/BLAS := mkl/g" Makefile.config && \
    pip2 install -r /opt/work/caffe/python/requirements.txt && \
    ln -s /usr/include/python2.7/ /usr/local/include/python2.7 && \
    ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/ /usr/local/include/python2.7/numpy && \
    make all -j4 && make pycaffe && make distribute && \
    make test && make runtest && \
    cd /opt/work && \
    ln /dev/null /dev/raw1394

#torch
RUN git clone https://github.com/torch/distro.git /opt/work/torch --recursive && \
    cd /opt/work/torch && git checkout ${TORCH_VERSION} && \
    echo "Acquire::http::Proxy \"${http_proxy}\";\nAcquire::https::Proxy \"${https_proxy}\";" > /etc/apt/apt.conf && \
    export CMAKE_INCLUDE_PATH=/opt/intel/mkl/include && \
    export CMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/opt/intel/mkl/compiler/lib/intel64 && \
    export LD_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$LD_LIBRARY_PATH && \
    ./install-deps && \
    yes | ./install.sh && \
    cd /opt/work/

#torch-rnn
RUN git config --global url.https://github.com/.insteadOf git://github.com/ && \
    cat ~/.gitconfig && \
    #luarocks install torch && \
    #luarocks install nn && \
    #luarocks install dpnn && \
    luarocks install torchx && \
    git clone https://github.com/Element-Research/dpnn.git && \
    cd dpnn && \
    git reset --hard ca0e99fa6d0dfd971f0f284520562765ae0865fd && \
    luarocks make rocks/dpnn-scm-1.rockspec && \
    #luarocks install torchx && \
    cd /opt/work && \
    git clone https://github.com/Element-Research/rnn.git && \
    cd rnn && \
    luarocks make rocks/rnn-scm-1.rockspec && \
    cd /opt/work

CMD ["/opt/work/jenkins/run-slave.sh"]

