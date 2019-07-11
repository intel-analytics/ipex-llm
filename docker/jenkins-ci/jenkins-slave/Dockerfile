FROM ubuntu:16.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ENV JAVA_7_HOME         /opt/work/jdk7
ENV JAVA_8_HOME         /opt/work/jdk8
ENV JAVA_HOME           ${JAVA_8_HOME}
ENV SCALA_HOME          /opt/work/scala
ENV CONDA_HOME          /opt/work/conda
ENV JENKINS_HOME        /opt/work/jenkins
ENV SPARK_2_1_HOME      /opt/work/spark-2.1.3
ENV SPARK_2_2_HOME      /opt/work/spark-2.2.2
ENV SPARK_2_3_HOME      /opt/work/spark-2.3.2
ENV SPARK_2_4_HOME      /opt/work/spark-2.4.0
ENV PATH                $SCALA_HOME/bin:${JAVA_HOME}/bin:${CONDA_HOME}/bin:${PATH}
ENV TORCH_VERSION       d03a42834bb1b674495b0c42de1716b66cc388f1
ENV TORCH_HOME		/opt/work/torch
ENV CAFFE_HOME      	/opt/work/caffe
ENV PATH                ${TORCH_HOME}/install/bin:${JAVA_HOME}/bin:${CONDA_HOME}/bin:${PATH}
ENV LANG                en_US.UTF-8
ENV LC_ALL              en_US.UTF-8

ADD ./run-slave.sh      /opt/work/jenkins/run-slave.sh
ADD ./slave.groovy      /opt/work/jenkins/slave.groovy

RUN apt-get update --fix-missing && \
    apt-get install -y vim curl nano wget unzip maven git bzip2 && \
    apt-get install -y locales && locale-gen en_US.UTF-8 && \
    apt-get install -y build-essential && \
    apt-get install -y protobuf-compiler libprotoc-dev && \
    apt-get install -y libgtk2.0-dev

#jdk8
RUN wget https://build.funtoo.org/distfiles/oracle-java/jdk-8u152-linux-x64.tar.gz && \
    gunzip jdk-8u152-linux-x64.tar.gz && \
    tar -xf jdk-8u152-linux-x64.tar -C /opt && \
    rm jdk-8u152-linux-x64.tar && \
    ln -s /opt/jdk1.8.0_152 ${JAVA_8_HOME} && \
#jdk7
    wget https://build.funtoo.org/distfiles/oracle-java/jdk-7u80-linux-x64.tar.gz && \
    gunzip jdk-7u80-linux-x64.tar.gz && \
    tar -xf jdk-7u80-linux-x64.tar -C /opt && \
    rm jdk-7u80-linux-x64.tar && \
    ln -s /opt/jdk1.7.0_80 ${JAVA_7_HOME} && \
#scala
    wget https://downloads.lightbend.com/scala/2.11.8/scala-2.11.8.tgz && \
    gunzip scala-2.11.8.tgz && \
    tar -xf scala-2.11.8.tar -C /opt && \
    rm scala-2.11.8.tar && \
    ln -s /opt/scala-2.11.8 ${SCALA_HOME} && \
#Jenkins
    chmod a+x /opt/work/jenkins/run-slave.sh && \
    chmod a+x /opt/work/jenkins/slave.groovy && \
    wget http://repo.jenkins-ci.org/releases/org/jenkins-ci/main/remoting/3.14/remoting-3.14.jar && \
    mv remoting-3.14.jar ${JENKINS_HOME} && \
#spark 2.1.3 2.2.2 2.3.2 2.4.0
    wget http://archive.apache.org/dist/spark/spark-2.1.3/spark-2.1.3-bin-hadoop2.7.tgz && \
    tar -xf spark-2.1.3-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.1.3-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.1.3-bin-hadoop2.7 ${SPARK_2_1_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-2.2.2/spark-2.2.2-bin-hadoop2.7.tgz && \
    tar -xf spark-2.2.2-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.2.2-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.2.2-bin-hadoop2.7 ${SPARK_2_2_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-2.3.2/spark-2.3.2-bin-hadoop2.7.tgz && \
    tar -xf spark-2.3.2-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.3.2-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.3.2-bin-hadoop2.7 ${SPARK_2_3_HOME} && \
    wget http://archive.apache.org/dist/spark/spark-2.4.0/spark-2.4.0-bin-hadoop2.7.tgz && \
    tar -xf spark-2.4.0-bin-hadoop2.7.tgz -C /opt/work && \
    rm spark-2.4.0-bin-hadoop2.7.tgz && \
    ln -s /opt/work/spark-2.4.0-bin-hadoop2.7 ${SPARK_2_4_HOME} && \
#cmake
    wget https://cmake.org/files/v3.12/cmake-3.12.1.tar.gz && \
    tar xf cmake-3.12.1.tar.gz && \
    cd cmake-3.12.1 && \
    ./configure && \
    make -j16 && \
    make install && \
    cd .. && \
#python-conda
    wget https://repo.continuum.io/miniconda/Miniconda3-4.3.31-Linux-x86_64.sh && \
    /bin/bash Miniconda3-4.3.31-Linux-x86_64.sh -f -b -p ${CONDA_HOME} && \
    rm Miniconda3-4.3.31-Linux-x86_64.sh && \
    conda config --system --prepend channels conda-forge && \
    conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    conda update --all --quiet --yes && \
    conda clean -tipsy

#TENSORFLOW MODELS
ENV PYTHONPATH /opt/work/models/research/slim:${PYTHONPATH}
RUN git clone https://github.com/tensorflow/models.git

#keras=1.2.2
RUN export TENSOR_FLOW_VERION=1.10.0 && \
    export KERAS_VERSION=1.2.2 && \
#py27
    conda create -y -n py27 python=2.7 && \
    conda install -y -n py27 -c conda-forge \
        mkdocs==1.0.4 scipy==1.2.1 pandas==0.24.2 scikit-learn==0.20.3 \
        matplotlib==2.2.3 seaborn==0.9.0 jupyter==1.0.0 plotly==3.9.0 \
        nltk==3.2.5 twine==1.13.0 pytest==4.5.0 pytest-xdist==1.28.0 \
        h5py==2.9.0 moviepy==0.2.3.5 scikit-image==0.14.3 \
        typing==3.6.6 numpy==1.16.3 pyyaml==5.1 \
        mkl==2019.3 mkl-include==2019.3 mkl-service==1.1.2 setuptools==41.0.1 \
        cmake==3.14.3 cffi==1.12.3 robotframework==3.1.1 requests==2.21.0 \
        networkx==2.2 tensorboard==1.10.0 libprotobuf==3.7.1 protobuf==3.7.1 \
        tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION&& \
    conda install -y -n py27 --channel https://conda.anaconda.org/menpo opencv3==3.1.0 && \
    conda install -y -n py27 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py27 -c pytorch pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 && \
    /bin/bash -c "source activate py27 && pip install onnx==1.3.0 && source deactivate" && \
#py35
    conda create -y -n py35 python=3.5 && \
    conda install -y -n py35 -c conda-forge \
        mkdocs==1.0.4 scipy==1.1.0 pandas==0.23.4 scikit-learn==0.20.0 \
        matplotlib==3.0.0 seaborn==0.9.0 jupyter==1.0.0 plotly==3.9.0 \
        nltk==3.2.5 twine==1.13.0 pytest==3.8.1 pytest-xdist==1.27.0 \
        h5py==2.8.0 moviepy==0.2.3.5 imageio-ffmpeg==0.3.0 \
        typing==3.6.6 numpy==1.15.2 pyyaml==3.13 \
        mkl==2019.3 mkl-include==2019.3 mkl-service==1.1.2 setuptools==40.4.3 \
        cmake==3.14.3 cffi==1.11.5 robotframework==3.1.1 requests==2.19.1 \
        networkx==2.2 tensorboard==1.10.0 libprotobuf==3.6.0 protobuf==3.6.0 \
        tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py35 -c anaconda protobuf==3.6.0 && \
    conda install -y -n py35 --channel https://conda.anaconda.org/menpo opencv3==3.1.0 && \
    conda install -y -n py35 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py35 -c pytorch pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 && \
    /bin/bash -c "source activate py35 && pip install onnx==1.3.0 && \
        pip install ray==0.7.0 && \
        pip install conda-pack==0.3.1 && \
        pip install psutil==5.6.2 && \
        pip install aiohttp==3.5.4 && \
        pip install setproctitle==1.1.10 && \
        source deactivate" && \
#py36
    conda create -y -n py36 python=3.6 && \
    conda install -y -n py36 -c conda-forge \
        mkdocs==1.0.4 scipy==1.2.1 pandas==0.24.2 scikit-learn==0.20.3 \
        matplotlib==3.0.3 seaborn==0.9.0 jupyter==1.0.0  plotly==3.9.0 \
        nltk==3.2.5 twine==1.13.0 pytest==4.5.0 pytest-xdist==1.28.0 \
        h5py==2.9.0 moviepy==0.2.3.5 imageio-ffmpeg==0.3.0 \
        typing==3.6.4 numpy==1.16.3 pyyaml==5.1 \
        mkl==2019.3 mkl-include==2019.3 mkl-service==1.1.2 setuptools==41.0.1 \
        cmake==3.14.3 cffi==1.12.3 robotframework==3.1.1 requests==2.21.0 \
        networkx==2.2 tensorboard==1.10.0 libprotobuf==3.7.1 protobuf==3.7.1 \
        tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py36 -c anaconda protobuf==3.7.1 && \
    conda install -y -n py36 --channel https://conda.anaconda.org/menpo opencv3==3.1.0 && \
    conda install -y -n py36 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py36 -c pytorch pytorch-cpu==1.0.0 torchvision-cpu==0.2.1 && \
    /bin/bash -c "source activate py36 && pip install onnx==1.3.0 && \
        pip install ray==0.7.0 && \
        pip install conda-pack==0.3.1 && \
        pip install psutil==5.6.2 && \
        pip install aiohttp==3.5.4 && \
        pip install setproctitle==1.1.10 && \
        source deactivate" && \
#keras=2.1.6
    export TENSOR_FLOW_VERION=1.10.0 && \
    export KERAS_VERSION=2.1.6 && \
#py27k216
    conda create -y -n py27k216 python=2.7 mkdocs==1.0.4 scipy==1.2.1 pandas==0.24.2 scikit-learn==0.20.3 matplotlib==2.2.3 seaborn==0.9.0 jupyter==1.0.0 && \
    conda install -y -n py27k216 plotly==3.9.0 nltk==3.2.5 twine==1.13.0 pytest==4.5.0 pytest-xdist==1.28.0 h5py==2.8.0 moviepy==0.2.3.5 && \
    conda install -y -n py27k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py27k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py27k216 typing==3.6.6 protobuf==3.7.1 numpy==1.16.3 pyyaml==5.1 mkl==2019.3 mkl-include==2019.3 setuptools==41.0.1 cmake==3.14.3 cffi==1.12.3 && \
    conda install -y -n py27k216 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py27k216 -c conda-forge robotframework==3.1.1 requests==2.21.0 && \
#py35k216
    conda create -y -n py35k216 python=3.5 mkdocs==1.0.4 scipy==1.1.0 pandas==0.23.4 scikit-learn==0.20.0 matplotlib==2.2.2 seaborn==0.9.0 jupyter==1.0.0 && \
    conda install -y -n py35k216 plotly==3.9.0 nltk==3.2.5 twine==1.13.0 pytest==3.8.1 pytest-xdist==1.27.0 h5py==2.8.0 moviepy==0.2.3.5 && \
    conda install -y -n py35k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py35k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py35k216 typing==3.6.6 protobuf==3.6.0 numpy==1.15.2 pyyaml==3.13 mkl==2019.3 mkl-include==2019.3 setuptools==40.4.3 cmake==3.14.3 cffi==1.11.5 && \
    conda install -y -n py35k216 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py35k216 -c conda-forge robotframework==3.1.1 requests==2.19.1 && \
#py36k216
    conda create -y -n py36k216 python=3.6 mkdocs==1.0.4 scipy==1.2.1 pandas==0.24.2 scikit-learn==0.20.3 matplotlib==3.0.3 seaborn==0.9.0 jupyter==1.0.0 && \
    conda install -y -n py36k216 plotly==3.9.0 nltk==3.2.5 twine==1.13.0 pytest==4.5.0 pytest-xdist==1.28.0 h5py==2.8.0 moviepy==0.2.3.5 && \
    conda install -y -n py36k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py36k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py36k216 typing==3.6.4 protobuf==3.7.1 numpy==1.16.3 pyyaml==5.1 mkl==2019.3 mkl-include==2019.3 setuptools==41.0.1 cmake==3.14.3 cffi==1.12.3 && \
    conda install -y -n py36k216 -c mingfeima mkldnn==0.16.1 && \
    conda install -y -n py36k216 -c conda-forge robotframework==3.1.1 requests==2.21.0

#torch
RUN git clone https://github.com/torch/distro.git /opt/work/torch --recursive && \
    cd /opt/work/torch && git checkout ${TORCH_VERSION} && \
    echo "Acquire::http::Proxy \"${http_proxy}\";\nAcquire::https::Proxy \"${https_proxy}\";" > /etc/apt/apt.conf && \
    export CMAKE_INCLUDE_PATH=/opt/intel/mkl/include && \
    export CMAKE_LIBRARY_PATH=/opt/intel/mkl/lib/intel64:/opt/intel/mkl/compiler/lib/intel64 && \
    export LD_LIBRARY_PATH=$CMAKE_LIBRARY_PATH:$LD_LIBRARY_PATH && \
    sed -i "s/sudo/ /g" install-deps && \
    ./install-deps && \
    yes | ./install.sh && \
    cd /opt/work/

#torch-rnn
RUN git config --global url.https://github.com/.insteadOf git://github.com/ && \
    cat ~/.gitconfig && \
    /opt/work/torch/install/bin/luarocks install torchx && \
    git clone https://github.com/Element-Research/dpnn.git && \
    cd dpnn && \
    git reset --hard ca0e99fa6d0dfd971f0f284520562765ae0865fd && \
    luarocks make rocks/dpnn-scm-1.rockspec && \
    cd /opt/work && \
    git clone https://github.com/Element-Research/rnn.git && \
    cd rnn && \
    luarocks make rocks/rnn-scm-1.rockspec && \
    cd /opt/work

#Caffe prepare
RUN apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler && \
    apt-get install -y libopencv-dev libopencv-contrib-dev && \
    apt-get install -y --no-install-recommends libboost-all-dev && \
    apt-get install -y libatlas-base-dev && \
    apt-get install -y python-dev python-pip gfortran-7 && \
    apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev && \
    apt-get install -y --no-install-recommends build-essential tcl tk gettext autoconf zlib1g-dev gcc && \
    apt-get install -y --no-install-recommends python-dev python3-dev libcurl4-openssl-dev 

#Caffe
RUN git clone https://github.com/BVLC/caffe.git && \
    cd /opt/work/caffe && \
    cp /opt/work/caffe/Makefile.config.example Makefile.config && \
    echo "CPU_ONLY := 1" >> Makefile.config && \
    export ANACONDA_HOME=/opt/work/conda/envs/py27 && \
    sed -i '71s/^[^#]/#&/' Makefile.config && \
    sed -i '76s/^#//' Makefile.config && \
    sed -i 's/# $(ANA/$(ANA/g' Makefile.config && \
    sed -i '86s/^[^#]/#&/' Makefile.config && \
    sed -i '87s/^#//' Makefile.config && \
    sed -i 's/LIBRARIES += opencv_core opencv_highgui opencv_imgproc/LIBRARIES += opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs/g' Makefile && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt update && \
    apt install g++-7 -y && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \
    update-alternatives --config gcc && \
    /bin/bash -c "source activate py27 && \
        easy_install -U pytest && \
        pip install -r /opt/work/caffe/python/requirements.txt && \
        ln -s /opt/work/conda/envs/py27/lib/python2.7 /usr/local/include/python2.7 && \
        ln -s /opt/work/conda/envs/py27/lib/python2.7/site-packages/numpy/core/include/numpy /usr/local/include/python2.7/numpy && \
        make all -j4 && make pycaffe && make distribute && \
        make test && \
        cd /opt/work && \
        export PYTHONPATH=${CAFFE_HOME}/python:${CAFFE_HOME}/distribute/python:${PYTHONPATH} && \
        ln /dev/null /dev/raw1394 && \
        source deactivate"

CMD ["/opt/work/jenkins/run-slave.sh"]
