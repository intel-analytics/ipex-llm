FROM ubuntu:16.04

MAINTAINER The Analytics-Zoo Authors https://github.com/intel-analytics/analytics-zoo

WORKDIR /opt/work

ENV JAVA_7_HOME		/opt/work/jdk7
ENV JAVA_8_HOME		/opt/work/jdk8
ENV JAVA_HOME		${JAVA_8_HOME}
ENV SCALA_HOME		/opt/work/scala
ENV CONDA_HOME		/opt/work/conda
ENV JENKINS_HOME	/opt/work/jenkins
ENV SPARK_1_6_HOME      /opt/work/spark-1.6.3
ENV SPARK_2_1_HOME      /opt/work/spark-2.1.3
ENV SPARK_2_2_HOME      /opt/work/spark-2.2.2
ENV SPARK_2_3_HOME      /opt/work/spark-2.3.2
ENV SPARK_2_4_HOME      /opt/work/spark-2.4.0
ENV PATH                $SCALA_HOME/bin:${JAVA_HOME}/bin:${CONDA_HOME}/bin:${PATH}
ENV LANG 		en_US.UTF-8
ENV LC_ALL 		en_US.UTF-8

ADD ./run-slave.sh 	/opt/work/jenkins/run-slave.sh
ADD ./slave.groovy 	/opt/work/jenkins/slave.groovy

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
#spark 1.6.3 2.1.3 2.2.2 2.3.2 2.4.0
    wget http://archive.apache.org/dist/spark/spark-1.6.3/spark-1.6.3-bin-hadoop2.6.tgz && \
    tar -xf spark-1.6.3-bin-hadoop2.6.tgz -C /opt/work && \
    rm spark-1.6.3-bin-hadoop2.6.tgz && \
    ln -s /opt/work/spark-1.6.3-bin-hadoop2.6 ${SPARK_1_6_HOME} && \
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
    conda install -y -n py27 -c conda-forge mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter \
        plotly nltk twine pytest pytest-xdist h5py moviepy \
        libprotobuf tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION \
        typing protobuf numpy pyyaml mkl mkl-include mkl-service setuptools \
        cmake cffi robotframework requests networkx==2.2 tensorboard && \
    conda install -y -n py27 --channel https://conda.anaconda.org/menpo opencv3 && \
    conda install -y -n py27 -c mingfeima mkldnn && \
    conda install -y -n py27 -c pytorch pytorch-cpu=1.0.0 torchvision-cpu=0.2.1 && \
    /bin/bash -c "source activate py27 && pip install onnx==1.3.0 && source deactivate" && \
#py35
    conda create -y -n py35 python=3.5 && \
    conda install -y -n py35 -c conda-forge mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter \
        plotly nltk twine pytest pytest-xdist h5py moviepy imageio-ffmpeg \
        typing numpy pyyaml mkl mkl-include mkl-service setuptools \
        cmake cffi robotframework requests networkx==2.2 tensorboard \
        libprotobuf protobuf tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py35 -c anaconda protobuf && \
    conda install -y -n py35 --channel https://conda.anaconda.org/menpo opencv3 && \
    conda install -y -n py35 -c mingfeima mkldnn && \
    conda install -y -n py35 -c pytorch pytorch-cpu=1.0.0 torchvision-cpu=0.2.1 && \
    /bin/bash -c "source activate py35 && pip install onnx==1.3.0 && source deactivate" && \
#py36
    conda create -y -n py36 python=3.6 && \
    conda install -y -n py36 -c conda-forge mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter \
        plotly nltk twine pytest pytest-xdist h5py moviepy imageio-ffmpeg \
        typing numpy pyyaml mkl mkl-include mkl-service setuptools \
        cmake cffi robotframework requests networkx==2.2 tensorboard \
        libprotobuf protobuf tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py36 -c anaconda protobuf && \
    conda install -y -n py36 --channel https://conda.anaconda.org/menpo opencv3 && \
    conda install -y -n py36 -c mingfeima mkldnn && \
    conda install -y -n py36 -c pytorch pytorch-cpu=1.0.0 torchvision-cpu=0.2.1 && \
    /bin/bash -c "source activate py36 && pip install onnx==1.3.0 && source deactivate" && \
#keras=2.1.6
    export TENSOR_FLOW_VERION=1.10.0 && \
    export KERAS_VERSION=2.1.6 && \
#py27k216
    conda create -y -n py27k216 python=2.7 mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter && \
    conda install -y -n py27k216 plotly nltk twine pytest pytest-xdist h5py moviepy && \
    conda install -y -n py27k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py27k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py27k216 typing protobuf numpy pyyaml mkl mkl-include setuptools cmake cffi && \
    conda install -y -n py27k216 -c mingfeima mkldnn && \
    conda install -y -n py27k216 -c conda-forge robotframework requests && \
#py35k216
    conda create -y -n py35k216 python=3.5 mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter && \
    conda install -y -n py35k216 plotly nltk twine pytest pytest-xdist h5py moviepy && \
    conda install -y -n py35k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py35k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py35k216 typing protobuf numpy pyyaml mkl mkl-include setuptools cmake cffi && \
    conda install -y -n py35k216 -c mingfeima mkldnn && \
    conda install -y -n py35k216 -c conda-forge robotframework requests && \
#py36k216
    conda create -y -n py36k216 python=3.6 mkdocs scipy pandas scikit-learn matplotlib seaborn jupyter && \
    conda install -y -n py36k216 plotly nltk twine pytest pytest-xdist h5py moviepy && \
    conda install -y -n py36k216 -c conda-forge tensorflow==$TENSOR_FLOW_VERION keras==$KERAS_VERSION && \
    conda install -y -n py36k216 -c conda-forge opencv==3.4.1 && \
    conda install -y -n py36k216 typing protobuf numpy pyyaml mkl mkl-include setuptools cmake cffi && \
    conda install -y -n py36k216 -c mingfeima mkldnn && \
    conda install -y -n py36k216 -c conda-forge robotframework requests

CMD ["/opt/work/jenkins/run-slave.sh"]
