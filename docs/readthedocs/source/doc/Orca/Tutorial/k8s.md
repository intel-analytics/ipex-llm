# Run on Kubernetes Clusters

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Kubernetes (K8s) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/docs/docs/tutorials/tutorial_example/Fashion_MNIST/) as a working example.


# 1. Key Concepts
## 1.1 Init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode, master, container_image, 
                  cores, memory, num_nodes, driver_cores, driver_memory, 
                  extra_python_lib, penv_archive, jars, conf)
```

In `init_orca_context`, you may specify necessary runtime configurations for running the example on K8s, including:
* `cluster_mode`: a String that specifies the underlying cluster; valid value includes `"local"`, `"yarn-client"`, `"yarn-cluster"`, __`"k8s-client"`__, __`"k8s-cluster"`__, `"bigdl-submit"`, `"spark-submit"`, etc.
* `master`: a URL format to specify the master address of K8s cluster.
* `container_image`: a String that specifies the name of docker container image for executors.
* `cores`: an Integer that specifies the number of cores for each executor (default to be `2`).
* `memory`: a String that specifies the memory for each executor (default to be `"2g"`).
* `num_nodes`: an Integer that specifies the number of executors (default to be `1`).
* `driver_cores`: an Integer that specifies the number of cores for the driver node (default to be `4`).
* `driver_memory`: a String that specifies the memory for the driver node (default to be `"1g"`).
* `extra_python_lib`: a String that specifies the path to extra Python packages, one of `.py`, `.zip` or `.egg` files (default to be `None`).
* `penv_archive`: a String that specifies the path to a packed Conda archive (default to be `None`).
* `jars`: a String that specifies the path to needed jars files (default to be `None`).
* `conf`: a Key-Value format to append extra conf for Spark (default to be `None`).

__Note__: 
* All arguments __except__ `cluster_mode` will be ignored when using `spark-submit` and Kubernetes deployment to submit and run Orca programs.

After the Orca programs finish, you should call `stop_orca_context` at the end of the program to release resources and shutdown the underlying distributed runtime engine (such as Spark or Ray).
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html).


## 1.2 K8s Client&Cluster
The difference between k8s-client and k8s-cluster is where you run your Spark driver. 

For k8s-client, the Spark driver runs in the client process (outside the K8s cluster), while for k8s-cluster the Spark driver runs inside the K8s cluster.

Please see more details in [K8s-Cluster](https://spark.apache.org/docs/latest/running-on-kubernetes.html#cluster-mode) and [K8s-Client](https://spark.apache.org/docs/latest/running-on-kubernetes.html#client-mode).



## 1.3 Load Data from Network File Systems (NFS)
When you are running programs on K8s, please load data from volumes and we use NFS in this tutorial as an example.

After mounting the volume (NFS) into BigDL container (see __[Section 2.2](#22-create-a-k8s-client-container)__), the Fashion-MNIST example could load data from NFS.

```python
import torch
import torchvision
import torchvision.transforms as transforms
from bigdl.orca.data.file import get_remote_file_to_local

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    get_remote_file_to_local(remote_path="/path/to/nfsdata", local_path="/tmp/dataset")

    trainset = torchvision.datasets.FashionMNIST(root="/bigdl/nfsdata/dataset", train=True, 
                                                 download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader
```



# 2. Create & Lunch BigDL K8s Container 
## 2.1 Pull Docker Image
Please pull the BigDL 2.1.0 `bigdl-k8s` image from [Docker Hub](https://hub.docker.com/r/intelanalytics/bigdl-k8s/tags) as follows:
```bash
sudo docker pull intelanalytics/bigdl-k8s:2.1.0
```

__Note:__
* If you need the nightly built BigDL, please pull the latest image as below:
    ```bash
    sudo docker pull intelanalytics/bigdl-k8s:latest
    ```
* The 2.1.0 and latest BigDL image is built on top of Spark 3.1.2.


## 2.2 Create a K8s Client Container
Please launch the __Client Container__ following the script below:
```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    intelanalytics/bigdl-k8s:2.1.0 bash
```

In the script:
* `--net=host`: use the host network stack for the Docker container;
* `-v /etc/kubernetes:/etc/kubernetes`: specify the path of kubernetes configurations;
* `-v /root/.kube:/root/.kube`: specify the path of kubernetes installation;

__Notes:__
* Please switch the tag from `2.1.0` to `latest` if you pull the latest BigDL image.
* The __Client Container__ contains all the required environment except K8s configs.
* You needn't to create an __Executor Container__ manually, which is scheduled by K8s at runtime.

We recommend you to specify more arguments when creating a container:
```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v /path/to/nfsdata:/bigdl/nfsdata \
    -e NOTEBOOK_PORT=12345 \
    -e NOTEBOOK_TOKEN="your-token" \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/bigdl-k8s:2.1.0 \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=nfsvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=2 \
    -e RUNTIME_EXECUTOR_CORES=2 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    intelanalytics/bigdl-k8s:2.1.0 bash 
```

__Notes:__ 
* Please make sure you are mounting the correct volumn path (e.g. NFS) in a container.
* Please switch the `2.1.0` tag to `latest` if you pull the latest BigDL image.

In the script:
* `--net=host`: use the host network stack for the Docker container;
* `/etc/kubernetes:/etc/kubernetes`: specify the path of kubernetes configurations;
* `/root/.kube:/root/.kube`: specify the path of kubernetes installation;
* `/path/to/nfsdata:/bigdl/data`: mount NFS path on host in a container as the sepcified path in value; 
* `NOTEBOOK_PORT`: an Integer that specifies port number for Notebook (only required by notebook);
* `NOTEBOOK_TOKEN`: a String that specifies the token for Notebook (only required by notebook);
* `RUNTIME_SPARK_MASTER`: a URL format that specifies the Spark master;
* `RUNTIME_K8S_SERVICE_ACCOUNT`: a String that specifies the service account for driver pod;
* `RUNTIME_K8S_SPARK_IMAGE`: the lanuched k8s image;
* `RUNTIME_PERSISTENT_VOLUME_CLAIM`: a String that specifies the Kubernetes volumeName;
* `RUNTIME_DRIVER_HOST`: a URL format that specifies the driver localhost (only required by client mode);
* `RUNTIME_DRIVER_PORT`: a String that specifies the driver port (only required by client mode);
* `RUNTIME_EXECUTOR_INSTANCES`: an Integer that specifies the number of executors;
* `RUNTIME_EXECUTOR_CORES`: an Integer that specifies the number of cores for each executor;
* `RUNTIME_EXECUTOR_MEMORY`: a String that specifies the memory for each executor;
* `RUNTIME_TOTAL_EXECUTOR_CORES`: an Integer that specifies the number of cores for all executors;
* `RUNTIME_DRIVER_CORES`: an Integer that specifies the number of cores for the driver node;
* `RUNTIME_DRIVER_MEMORY`: a String that specifies the memory for the driver node;


## 2.3 Launch the K8s Client Container
Once the container is created, docker image would return a `containerID`, please launch the container following the command below:
```bash
sudo docker exec -it <containerID> bash
```



# 3. Prepare Environment
In the launched BigDL K8s Container, please setup environment following steps below:
## 3.1 Install Python Libraries
### 3.1.1 Install Conda
Please use conda to prepare the Python environment on the __Client Container__ (where you submit applications), you could download and install Conda following [Conda User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) or executing the command as below.

```bash
# Download Anaconda installation script 
wget -P /tmp https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

# Execute the script to install conda
bash /tmp/Anaconda3-2020.02-Linux-x86_64.sh

# Please type this command in your terminal to activate Conda environment
source ~/.bashrc
```

### 3.1.2 Use Conda to Install BigDL and Other Python Libraries
Create a Conda environment, install BigDL and all needed Python libraries in the activate Conda:
```bash
# "env" is conda environment name, you can use any name you like.
# Please change Python version to 3.8 if you need a Python 3.8 environment.
conda create -n env python=3.7 
conda activate env
```

Please install the 2.1.0 release version of BigDL (built on top of Spark 3.1.2) as follows:
```bash
pip install bigdl-spark3
```

When you are running in the latest BigDL image, please install the nightly build of BigDL as follows:
```bash
pip install --pre --upgrade bigdl-spark3
```

Please install torch and torchvision to run the Fashion-MNIST example:
```bash
pip install torch torchvision
```

__Notes:__
* Using Conda to install BigDL will automatically install libraries including `pyspark==3.1.2`, and etc.
* You can install BigDL Orca built on top of Spark 3.1.2 as follows:
    ```bash
    # Install the latest release version
    pip install bigdl-orca-spark3

    # Install the latest nightly build version
    pip install --pre --upgrade bigdl-spark3-orca

    # You need to install torch and torchvision manually
    pip install torch torchvision
    ```

    Installing bigdl-orca-spark3 will automatically install `pyspark==3.1.2`.

* You also need to install any additional python libraries that your application depends on in the Conda environment.

* It's only need for you to install needed Python libraries using Conda, since the BigDL K8s container has already setup `JAVA_HOME`, `BIGDL_HOME`, `SPARK_HOME`, `SPARK_VERSION`, etc.

Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).



# 4. Prepare Dataset
To run the example provided by this tutorial on K8s, you should upload the dataset to to a K8s volumn (e.g. NFS).

Please download the Fashion-MNIST dataset manually on your __Develop Node__ (where you launch the container image). 

```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

# Move the dataset to NFS
mv /path/to/fashion-mnist/data/fashion /bigdl/nfsdata/dataset/FashionMNIST/raw

# Extract FashionMNIST archives
gzip -dk /bigdl/nfsdata/dataset/FashionMNIST/raw/*
```

__Note:__ PyTorch requires tge directory of dataset where `FashionMNIST/raw/train-images-idx3-ubyte` and `FashionMNIST/raw/t10k-images-idx3-ubyte` exist.


# 5. Prepare Custom Modules
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`. 

The FasionMNIST example needs to import modules from `model.py`.

__Note:__ Please upload the extra Python dependency files to NFS when running the program on k8s-cluster mode (see more details in __[Section 6.2.2](#622-k8s-cluster)__).

* When using `python` command, please specify `extra_python_lib` in `init_orca_context`.
    ```python
    import os
    from bigdl.orca import init_orca_context, stop_orca_context
    from model import model_creator, optimizer_creator

    conf={
          "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
          "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
        }

    init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=2, memory="2g",
                      master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>",
                      container_image="intelanalytics/bigdl-k8s:latest",
                      extra_python_lib="/path/to/model.py", conf=conf)
    ```
    
    Please see more details in [Orca Document](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html#python-dependencies).

* When using `spark-submit` script, please specify `--py-files` option in the script.    
    ```bash
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,file:///bigdl/nfsdata/model.py
    ```

    Then import custom modules:
    ```python
    from bigdl.orca import init_orca_context, stop_orca_context
    from model import model_creator, optimizer_creator

    init_orca_context(cluster_mode="spark-submit")
    ```

    Please see more details in [Spark Document](https://spark.apache.org/docs/latest/submitting-applications.html). 

__Notes:__
* You could follow the steps below to use a zipped package instead (recommended if your program depends on a nested directory of Python files) :
    1. Compress the directory into a Zipped Package.
        ```bash
        zip -q -r FashionMNIST_zipped.zip FashionMNIST
        ```
    2. Please follow the same method as above (using `.py` files) to upload the zipped package (`FashionMNIST_zipped.zip`) to K8s. 
    3. You should import custom modules from the unzipped file as below.
        ```python
        from FashionMNIST.model import model_creator, optimizer_creator
        ```



# 6. Run Jobs on K8s
In the following part, we will show you how to submit and run the Orca example on K8s:
* Use `python` command
* Use `spark-submit` script
* Use Kubernetes Deployment (with Conda Archive)
* Use Kubernetes Deployment (with Integrated Image)

## 6.1 Use `python` command
### 6.1.1 K8s-Client
Before running the example on `k8s-client` mode, you should:
* On the __Client Container__:
    1. Please call `init_orca_context` at very begining part of each Orca program.
        ```python
        from bigdl.orca import init_orca_context, stop_orca_context

        conf={
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
            "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
            }

        init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=2, memory="2g",
                        master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>",
                        container_image="intelanalytics/bigdl-k8s:2.1.0",
                        extra_python_lib="/path/to/model.py", conf=conf)
        ```

        To load data from NFS, please use the following configuration propeties: 
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into executor pods;
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;
    2. Using Conda to install BigDL and needed Python dependency libraries (see __[Section 3](#3-prepare-environment)__).

Please run the Fashion-MNIST example following the command below:
```bash
python train.py --cluster_mode k8s-client --remote_dir file:///bigdl/nfsdata/dataset
```

In the script:
* `cluster_mode`: set the cluster_mode in `init_orca_context`.
* `remote_dir`: directory on NFS for loading the dataset.


### 6.1.2 K8s-Cluster
Before running the example on `k8s-cluster` mode, you should:
* On the __Client Container__:
    1. Please call `init_orca_context` at very begining part of each Orca program.
        ```python
        from bigdl.orca import init_orca_context, stop_orca_context

        conf={
              "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
              "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
              "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
              "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
              "spark.kubernetes.authenticate.driver.serviceAccountName":"spark",
              "spark.kubernetes.file.upload.path":"/bigdl/nfsdata/"
              }

        init_orca_context(cluster_mode="k8s-cluster", num_nodes=2, cores=2, memory="2g",
                          master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>", 
                          container_image="intelanalytics/bigdl-k8s:2.1.0",
                          penv_archive="file:///bigdl/nfsdata/environment.tar.gz",
                          extra_python_lib="/bigdl/nfsdata/model.py", conf=conf)
        ```

        When running Orca programs on `k8s-cluster` mode, please use the following additional configuration propeties: 
        * `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into driver pod;
        * `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to driver pod on the NFS path specified in value;
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into executor pods;
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;
        * `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for driver pod;
        * `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in cluster mode;
    2. Using Conda to install BigDL and needed Python dependency libraries (see __[Section 3](#3-prepare-environment)__), then pack the current activate Conda environment to an archive.
        ```
        conda pack -o /path/to/environment.tar.gz
        ```
* On the __Develop Node__:
    1. Upload the Conda archive to NFS.
        ```bash
        docker cp <containerID>:/opt/spark/work-dir/environment.tar.gz /bigdl/nfsdata
        ```
    2. Upload the example Python file to NFS.
        ```bash
        mv /path/to/train.py /bigdl/nfsdata
        ```
    3. Upload the extra Python dependency file to NFS.
        ```bash
        mv /path/to/model.py /bigdl/nfsdata
        ```

Please run the Fashion-MNIST example in __Client Container__ following the command below:
```bash
python /bigdl/nfsdata/train.py --cluster_mode k8s-cluster --remote_dir /bigdl/nfsdata/dataset
```

In the script:
* `cluster_mode`: set the cluster_mode in `init_orca_context`.
* `remote_dir`: directory on NFS for loading the dataset.

__Note:__ It will return a `driver pod name` when the application is completed.

Please retreive training stats on the __Develop Node__ following the command below:
* Retrive training logs on the driver pod:
    ```bash
    kubectl logs <driver-pod-name>
    ```

* Check pod status or get basic informations around pod:
    ```bash
    kubectl describe pod <driver-pod-name>
    ```


## 6.2 Use `spark-submit` Script
### 6.2.1 K8s Client
Before submitting the example on `k8s-client` mode, you should:
* On the __Client Container__:
    1. Please call `init_orca_context` at very begining part of each Orca program.
        ```python
        from bigdl.orca import init_orca_context

        init_orca_context(cluster_mode="spark-submit")
        ```
    2. Using Conda to install BigDL and needed Python dependency libraries (see __[Section 3](#3-prepare-environment)__), then pack the current activate Conda environment to an archive.
        ```bash
        conda pack -o environment.tar.gz
        ```

Please submit the example following the script below:
```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode client \
    --name orca-k8s-client-tutorial \
    --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
    --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
    --driver-cores ${RUNTIME_DRIVER_CORES} \
    --driver-memory ${RUNTIME_DRIVER_MEMORY} \
    --executor-cores ${RUNTIME_EXECUTOR_CORES} \
    --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
    --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=./env/bin/python \
    --archives /path/to/environment.tar.gz#env \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_3.1.2-2.1.0-python-api.zip,/path/to/train.py,/path/to/model.py \
    --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    local:///path/to/train.py --cluster_mode "spark-submit" --remote_dir /bigdl/nfsdata/dataset
```

In the script:
* `master`: the spark master with a URL format;
* `deploy-mode`: set it to `client` when submitting in client mode;
* `name`: the name of Spark application;
* `spark.driver.host`: the localhost for driver pod (only required when submitting in client mode);
* `spark.kubernetes.container.image`: the BigDL docker image you downloaded; 
* `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for driver pod;
* `spark.pyspark.driver.python`: specify the Python location in Conda archive as driver's Python environment;
* `spark.pyspark.python`: specify the Python location in Conda archive as executors' Python environment;
* `archives`: upload the packed Conda archive to K8s;
* `properties-file`: upload BigDL configuration properties to K8s;
* `py-files`: upload extra Python dependency files to K8s;
* `spark.driver.extraClassPath`: upload and register the BigDL jars files to the driver's classpath;
* `spark.executor.extraClassPath`: upload and register the BigDL jars files to the executors' classpath;
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` with specified volumnName to mount `persistentVolume` into executor pods;
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: add specified volumeName of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;
* `cluster_mode`: the cluster_mode in `init_orca_context`;
* `remote_dir`: directory on NFS for loading the dataset.


### 6.2.2 K8s Cluster
Before submitting the example on `k8s-cluster` mode, you should:
* On the __Client Container__:
    1. Please call `init_orca_context` at very begining part of each Orca program.
        ```python
        from bigdl.orca import init_orca_context

        init_orca_context(cluster_mode="spark-submit")
        ```
    2. Using Conda to install BigDL and needed Python dependency libraries (see __[Section 3](#3-prepare-environment)__), then pack the Conda environment to an archive.
        ```bash
        conda pack -o environment.tar.gz
        ```

* On the __Develop Node__ (where you launch the __Client Container__):
    1. Upload Conda archive to NFS.
        ```bash
        docker cp <containerID>:/path/to/environment.tar.gz /bigdl/nfsdata
        ```

    2. Upload the example python files to NFS.
        ```bash
        mv /path/to/train.py /bigdl/nfsdata
        ```

    3. Upload the extra Python dependencies to NFS.
        ```bash
        mv /path/to/model.py /bigdl/nfsdata
        ```

Please run the example following the script below in the __Client Container__:
```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode cluster \
    --name orca-k8s-cluster-tutorial \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
    --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
    --archives file:///bigdl/nfsdata/environment.tar.gz#python_env \
    --conf spark.pyspark.python=python_env/bin/python \
    --conf spark.executorEnv.PYTHONHOME=python_env \
    --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata \
    --executor-cores ${RUNTIME_EXECUTOR_CORES} \
    --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
    --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
    --driver-cores ${RUNTIME_DRIVER_CORES} \
    --driver-memory ${RUNTIME_DRIVER_MEMORY} \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files local://${BIGDL_HOME}/python/bigdl-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip,file:///bigdl/nfsdata/train.py,file:///bigdl/nfsdata/model.py \
    --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    file:///bigdl/nfsdata/train.py --cluster_mode "spark-submit" --remote_dir /bigdl/nfsdata/dataset
```

In the script:
* `master`: the spark master with a URL format;
* `deploy-mode`: set it to `cluster` when submitting in cluster mode;
* `name`: the name of Spark application;
* `spark.kubernetes.container.image`: the BigDL docker image you downloaded; 
* `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for driver pod;
* `archives`: upload the Conda archive to K8s;
* `properties-file`: upload BigDL configuration properties to K8s;
* `py-files`: upload needed extra Python dependency files to K8s;
* `spark.pyspark.python`: specify the Python location in Conda archive as executors' Python environment;
* `spark.executorEnv.PYTHONHOME`: the search path of Python libraries on executor pod;
* `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in cluster mode;
* `spark.driver.extraClassPath`: upload and register the BigDL jars files to the driver's classpath;
* `spark.executor.extraClassPath`: upload and register the BigDL jars files to the executors' classpath;
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` with specified volumnName to mount `persistentVolume` into driver pod;
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: add specified volumeName of the volumeType `persistentVolumeClaim` to driver pod on the NFS path specified in value;
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` with specified volumnName to mount `persistentVolume` into executor pods;
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: add specified volumeName of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;
* `cluster_mode`: specify the cluster_mode in `init_orca_context`;
* `remote_dir`: directory on NFS for loading the dataset.


Please retrieve training stats on the __Develop Node__ following the commands below:
* Retrive training logs on the driver pod:
    ```bash
    kubectl logs `orca-k8s-cluster-tutorial-driver`
    ```

* Check pod status or get basic informations around pod using:
    ```bash
    kubectl describe pod `orca-k8s-cluster-tutorial-driver`
    ```


## 6.3 Use Kubernetes Deployment (with Conda Archive)
BigDL supports users (which want to execute programs directly on __Develop Node__) to run an application by creating a Kubernetes Deployment object.

Before submitting the Orca application, you should:
* On the __Develop Node__
    1. Use Conda to install BigDL and needed Python dependency libraries (see __[Section 3](#3-prepare-environment)__), then pack the activate Conda environment to an archive.
        ```bash
        conda pack -o environment.tar.gz
        ```
    2. Upload Conda archive, example Python files and extra Python dependencies to NFS.
        ```bash
        # Upload Conda archive
        cp /path/to/environment.tar.gz /bigdl/nfsdata

        # Upload example Python files
        cp /path/to/train.py /bigdl/nfsdata

        # Uplaod extra Python dependencies
        cp /path/to/model.py /bigdl/nfsdata
        ```

### 6.3.1 K8s Client
BigDL has provided an example YAML file (see __[orca-tutorial-client.yaml](../../../../../../python/orca/tutorial/pytorch/docker/orca-tutorial-client.yaml)__, which describes a Deployment that runs the `intelanalytics/bigdl-k8s:2.1.0` image) to run the tutorial FashionMNIST program on k8s-client mode:

__Notes:__ 
* Please call `init_orca_context` at very begining part of each Orca program.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="spark-submit")
    ```
* Spark client needs to specify `spark.pyspark.driver.python`, this python env should be on NFS dir.
    ```bash
    --conf spark.pyspark.driver.python=/bigdl/nfsdata/python_env/bin/python \
    ```

```bash
# orca-tutorial-client.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: orca-pytorch-job
spec:
  template:
    spec:
      serviceAccountName: spark
      restartPolicy: Never
      hostNetwork: true
      containers:
      - name: spark-k8s-client
        image: intelanalytics/bigdl-k8s:2.1.0
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh","-c"]
        args: ["
                export RUNTIME_DRIVER_HOST=$( hostname -I | awk '{print $1}' );
                ${SPARK_HOME}/bin/spark-submit \
                --master ${RUNTIME_SPARK_MASTER} \
                --deploy-mode ${SPARK_MODE} \
                --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
                --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
                --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
                --name orca-pytorch-tutorial \
                --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --archives file:///bigdl/nfsdata/environment.tar.gz#python_env \
                --conf spark.pyspark.driver.python=/bigdl/nfsdata/python_env/bin/python \
                --conf spark.pyspark.python=python_env/bin/python \
                --conf spark.executorEnv.PYTHONHOME=python_env \
                --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata/ \
                --num-executors 2 \
                --executor-cores 16 \
                --executor-memory 50g \
                --total-executor-cores 32 \
                --driver-cores 4 \
                --driver-memory 50g \
                --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
                --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///bigdl/nfsdata/train.py,local:///bigdl/nfsdata/model.py \
                --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
                --conf spark.kubernetes.executor.deleteOnTermination=True \
                --conf spark.sql.catalogImplementation='in-memory' \
                --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
                --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
                local:///bigdl/nfsdata/train.py
                --cluster_mode spark-submit
                --remote_dir file:///bigdl/nfsdata/dataset
                "]
        securityContext:
          privileged: true
        env:
          - name: RUNTIME_K8S_SPARK_IMAGE
            value: intelanalytics/bigdl-k8s:2.1.0
          - name: RUNTIME_SPARK_MASTER
            value: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>
          - name: RUNTIME_DRIVER_PORT
            value: !!str 54321
          - name: SPARK_MODE
            value: client
          - name: RUNTIME_K8S_SERVICE_ACCOUNT
            value: spark
          - name: BIGDL_HOME
            value: /opt/bigdl-2.1.0
          - name: SPARK_HOME
            value: /opt/spark
          - name: SPARK_VERSION
            value: 3.1.2
          - name: BIGDL_VERSION
            value: 2.1.0
        resources:
          requests:
            cpu: 1
          limits:
            cpu: 4
        volumeMounts:
          - name: nfs-storage
            mountPath: /bigdl/nfsdata
          - name: nfs-storage
            mountPath: /root/.kube/config
            subPath: kubeconfig
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: nfsvolumeclaim
```

In the YAML file:
* `metadata`: A nested object filed that every deployment object must specify a metadata.
    * `name`: A string that uniquely identifies this object and job.
* `restartPolicy`: Restart policy for all Containers within the pod. One of Always, OnFailure, Never. Default to Always.
* `containers`: A single application Container that you want to run within a pod.
    * `name`: Name of the Container, each Container in a pod must have a unique name.
    * `image`: Name of the Container image.
    * `imagePullPolicy`: Image pull policy. One of Always, Never and IfNotPresent. Defaults to Always if `:latest` tag is specified, or IfNotPresent otherwise.
    * `command`: command for the containers that run in the Pod.
    * `args`: arguments to submit the spark application in the Pod. See more details of the `spark-submit` script in __[Section 6.2.1](#621-k8s-client)__.
    * `securityContext`: SecurityContext defines the security options the container should be run with.
    * `env`: List of environment variables to set in the Container, which will be used when submitting the application.
        * `env.name`: Name of the environment variable.
        * `env.value`: Value of the environment variable.
    * `resources`: Allocate resources in the cluster to each pod.
        * `resource.limits`: Limits describes the maximum amount of compute resources allowed.
        * `resource.requests`: Requests describes the minimum amount of compute resources required.
    * `volumeMounts`: Declare where to mount volumes into containers.
        * `name`: Match with the Name of a Volume.
        * `mountPath`: Path within the Container at which the volume should be mounted.
        * `subPath`: Path within the volume from which the Container's volume should be mounted.
    * `volume`: specify the volumes to provide for the Pod.
        * `persistentVolumeClaim`: mount a PersistentVolume into a Pod

Create a Pod and run Fashion-MNIST application based on the YAML file.
```bash
kubectl apply -f orca-tutorial-client.yaml
```

List all pods to find the driver pod, which will be named as `orca-pytorch-job-xxx`.
```bash
# find out driver pod
kubectl get pods
```

View logs from the driver pod to retrive the training stats. 
```bash
# retrive training logs
kubectl logs `orca-pytorch-job-xxx`
```

After the task finish, you could delete the job as the command below.
```bash
kubectl delete job orca-pytorch-job
```

### 6.3.2 K8s Cluster
BigDL has provided an example YAML file (see __[orca-tutorial-cluster.yaml](../../../../../../python/orca/tutorial/pytorch/docker/orca-tutorial-cluster.yaml)__, which describes a Deployment that runs the `intelanalytics/bigdl-k8s:2.1.0` image) to run the tutorial FashionMNIST program on k8s-cluster mode:

__Notes:__ 
* Please call `init_orca_context` at very begining part of each Orca program.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="spark-submit")
    ```

```bash
# orca-tutorial-cluster.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: orca-pytorch-job
spec:
  template:
    spec:
      serviceAccountName: spark
      restartPolicy: Never
      hostNetwork: true
      containers:
      - name: spark-k8s-cluster
        image: intelanalytics/bigdl-k8s:2.1.0
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh","-c"]
        args: ["
                ${SPARK_HOME}/bin/spark-submit \
                --master ${RUNTIME_SPARK_MASTER} \
                --deploy-mode ${SPARK_MODE} \
                --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
                --name orca-pytorch-tutorial \
                --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --archives file:///bigdl/nfsdata/environment.tar.gz#python_env \
                --conf spark.pyspark.python=python_env/bin/python \
                --conf spark.executorEnv.PYTHONHOME=python_env \
                --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata/ \
                --num_executors 2 \
                --executor-cores 16 \
                --executor-memory 50g \
                --total-executor-cores 32 \
                --driver-cores 4 \
                --driver-memory 50g \
                --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
                --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///bigdl/nfsdata/train.py,local:///bigdl/nfsdata/model.py \
                --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
                --conf spark.kubernetes.executor.deleteOnTermination=True \
                --conf spark.sql.catalogImplementation='in-memory' \
                --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
                --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
                local:///bigdl/nfsdata/train.py
                --cluster_mode spark-submit
                --remote_dir file:///bigdl/nfsdata/dataset
                "]
        securityContext:
          privileged: true
        env:
          - name: RUNTIME_K8S_SPARK_IMAGE
            value: intelanalytics/bigdl-k8s:2.1.0
          - name: RUNTIME_SPARK_MASTER
            value: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>
          - name: SPARK_MODE
            value: cluster
          - name: RUNTIME_K8S_SERVICE_ACCOUNT
            value: spark
          - name: BIGDL_HOME
            value: /opt/bigdl-2.1.0
          - name: SPARK_HOME
            value: /opt/spark
          - name: SPARK_VERSION
            value: 3.1.2
          - name: BIGDL_VERSION
            value: 2.1.0
        resources:
          requests:
            cpu: 1
          limits:
            cpu: 4
        volumeMounts:
          - name: nfs-storage
            mountPath: /bigdl/nfsdata
          - name: nfs-storage
            mountPath: /root/.kube/config
            subPath: kubeconfig
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: nfsvolumeclaim
```

In the YAML file:
* `restartPolicy`: Restart policy for all Containers within the pod. One of Always, OnFailure, Never. Default to Always.
* `containers`: A single application Container that you want to run within a pod.
    * `name`: Name of the Container, each Container in a pod must have a unique name.
    * `image`: Name of the Container image.
    * `imagePullPolicy`: Image pull policy. One of Always, Never and IfNotPresent. Defaults to Always if `:latest` tag is specified, or IfNotPresent otherwise.
    * `command`: command for the containers that run in the Pod.
    * `args`: arguments to submit the spark application in the Pod. See more details of the `spark-submit` script in __[Section 6.2.2](#622-k8s-cluster)__.
    * `securityContext`: SecurityContext defines the security options the container should be run with.
    * `env`: List of environment variables to set in the Container, which will be used when submitting the application.
        * `env.name`: Name of the environment variable.
        * `env.value`: Value of the environment variable.
    * `resources`: Allocate resources in the cluster to each pod.
        * `resource.limits`: Limits describes the maximum amount of compute resources allowed.
        * `resource.requests`: Requests describes the minimum amount of compute resources required.
    * `volumeMounts`: Declare where to mount volumes into containers.
        * `name`: Match with the Name of a Volume.
        * `mountPath`: Path within the Container at which the volume should be mounted.
        * `subPath`: Path within the volume from which the Container's volume should be mounted.
    * `volume`: specify the volumes to provide for the Pod.
        * `persistentVolumeClaim`: mount a PersistentVolume into a Pod

Create a Pod and run Fashion-MNIST application based on the YAML file.
```bash
kubectl apply -f orca-tutorial-cluster.yaml
```

List all pods to find the driver pod (since the client pod only returns training status), which will be named as `orca-pytorch-job-driver`.
```bash
# checkout training status
kubectl logs `orca-pytorch-job-xxx`

# find out driver pod
kubectl get pods
```

View logs from the driver pod to retrive the training stats. 
```bash
# retrive training logs
kubectl logs `orca-pytorch-job-driver`
```

After the task finish, you could delete the job as the command below.
```bash
kubectl delete job orca-pytorch-job
```


## 6.4 Use Kubernetes Deployment (without Integrared Image)
BigDL also supports uses to skip preparing envionment through providing a container image (`intelanalytics/bigdl-k8s:orca-2.1.0`) which has integrated all BigDL required environments.

__Notes:__
* The image will be pulled automatically when you deploy pods with the YAML file.
* Conda archive is no longer needed in this method, please skip __[Section 3](#3-prepare-environment)__, since BigDL has integrated environment in `intelanalytics/bigdl-k8s:orca-2.1.0`. 
* If you need to install extra Python libraries which may not included in the image, please submit applications with Conda archive (refer to __[Section 6.3](#63-use-kubernetes-deployment)__).

Before submitting the example application, you should:
* On the __Develop Node__
    * Download dataset and upload it to NFS.
        ```bash
        mv /path/to/dataset /bigdl/nfsdata 
        ```
    * Upload example Python files and extra Python dependencies to NFS.
        ```bash
        # Upload example Python files
        cp /path/to/train.py /bigdl/nfsdata

        # Uplaod extra Python dependencies
        cp /path/to/model.py /bigdl/nfsdata
        ```

### 6.4.1 K8s Client
BigDL has provided an example YAML file (see __[integrated_image_client.yaml](../../../../../../python/orca/tutorial/pytorch/docker/integrate_image_client.yaml)__, which describes a deployment that runs the `intelanalytics/bigdl-k8s:orca-2.1.0` image) to run the tutorial FashionMNIST program on k8s-client mode:

__Notes:__
* Please call `init_orca_context` at very begining part of each Orca program.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="spark-submit")
    ```
* Spark client needs to specify `spark.pyspark.driver.python`, this python env should be on NFS dir.
    ```bash
    --conf spark.pyspark.driver.python=/bigdl/nfsdata/orca_env/bin/python \
    ```

```bash
#integrate_image_client.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: orca-integrate-job
spec:
  template:
    spec:
      serviceAccountName: spark
      restartPolicy: Never
      hostNetwork: true
      containers:
      - name: spark-k8s-client
        image: intelanalytics/bigdl-spark-3.1.2:orca-2.1.0
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh","-c"]
        args: ["
                export RUNTIME_DRIVER_HOST=$( hostname -I | awk '{print $1}' );
                ${SPARK_HOME}/bin/spark-submit \
                --master ${RUNTIME_SPARK_MASTER} \
                --deploy-mode ${SPARK_MODE} \
                --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
                --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
                --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
                --name orca-integrate-pod \
                --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
                --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata \
                --conf spark.pyspark.driver.python=python \
                --conf spark.pyspark.python=/usr/local/envs/bigdl/bin/python \
                --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata/ \
                --executor-cores 10 \
                --executor-memory 50g \
                --num-executors 4 \
                --total-executor-cores 40 \
                --driver-cores 10 \
                --driver-memory 50g \
                --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
                --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///bigdl/nfsdata/train.py,local:///bigdl/nfsdata/model.py \
                --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
                --conf spark.sql.catalogImplementation='in-memory' \
                --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
                --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
                local:///bigdl/nfsdata/train.py
                --cluster_mode spark-submit
                --remote_dir file:///bigdl/nfsdata/dataset
                "]
        securityContext:
          privileged: true
        env:
          - name: RUNTIME_K8S_SPARK_IMAGE
            value: intelanalytics/bigdl-spark-3.1.2:orca-2.1.0
          - name: RUNTIME_SPARK_MASTER
            value: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>
          - name: RUNTIME_DRIVER_PORT
            value: !!str 54321
          - name: SPARK_MODE
            value: client
          - name: RUNTIME_K8S_SERVICE_ACCOUNT
            value: spark
          - name: BIGDL_HOME
            value: /opt/bigdl-2.1.0
          - name: SPARK_HOME
            value: /opt/spark
          - name: SPARK_VERSION
            value: 3.1.2
          - name: BIGDL_VERSION
            value: 2.1.0
        resources:
          requests:
            cpu: 1
          limits:
            cpu: 4
        volumeMounts:
          - name: nfs-storage
            mountPath: /bigdl/nfsdata
          - name: nfs-storage
            mountPath: /root/.kube/config
            subPath: kubeconfig
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: nfsvolumeclaim
```

In the YAML file:
* `restartPolicy`: Restart policy for all Containers within the pod. One of Always, OnFailure, Never. Default to Always.
* `containers`: A single application Container that you want to run within a pod.
    * `name`: Name of the Container, each Container in a pod must have a unique name.
    * `image`: Name of the Container image.
    * `imagePullPolicy`: Image pull policy. One of Always, Never and IfNotPresent. Defaults to Always if `:latest` tag is specified, or IfNotPresent otherwise.
    * `command`: command for the containers that run in the Pod.
    * `args`: arguments to submit the spark application in the Pod. See more details of the `spark-submit` script in __[Section 6.2.1](#621-k8s-client)__.
    * `securityContext`: SecurityContext defines the security options the container should be run with.
    * `env`: List of environment variables to set in the Container, which will be used when submitting the application.
        * `env.name`: Name of the environment variable.
        * `env.value`: Value of the environment variable.
    * `resources`: Allocate resources in the cluster to each pod.
        * `resource.limits`: Limits describes the maximum amount of compute resources allowed.
        * `resource.requests`: Requests describes the minimum amount of compute resources required.
    * `volumeMounts`: Declare where to mount volumes into containers.
        * `name`: Match with the Name of a Volume.
        * `mountPath`: Path within the Container at which the volume should be mounted.
        * `subPath`: Path within the volume from which the Container's volume should be mounted.
    * `volume`: specify the volumes to provide for the Pod.
        * `persistentVolumeClaim`: mount a PersistentVolume into a Pod

Create a Pod and run Fashion-MNIST application based on the YAML file.
```bash
kubectl apply -f integrate_image_client.yaml
```

List all pods to find the driver pod, which will be named as `orca-integrate-job-xxx`.
```bash
# find out driver pod
kubectl get pods
```

View logs from the driver pod to retrive the training stats. 
```bash
# retrive training logs
kubectl logs `orca-integrate-job-xxx`
```

After the task finish, you could delete the job as the command below.
```bash
kubectl delete job orca-integrate-job
```

### 6.4.2 K8s Cluster
BigDL has provided an example YAML file (see __[integrate_image_cluster.yaml](../../../../../../python/orca/tutorial/pytorch/docker/integrate_image_cluster.yaml)__, which describes a deployment that runs the `intelanalytics/bigdl-k8s:orca-2.1.0` image) to run the tutorial FashionMNIST program on k8s-cluster mode:

__Notes:__
* Please call `init_orca_context` at very begining part of each Orca program.
    ```python
    from bigdl.orca import init_orca_context

    init_orca_context(cluster_mode="spark-submit")
    ```

```bash
# integrate_image_cluster.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: orca-integrate-job
spec:
  template:
    spec:
      serviceAccountName: spark
      restartPolicy: Never
      hostNetwork: true
      containers:
      - name: spark-k8s-cluster
        image: intelanalytics/bigdl-spark-3.1.2:orca-2.1.0
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh","-c"]
        args: ["
                ${SPARK_HOME}/bin/spark-submit \
                --master ${RUNTIME_SPARK_MASTER} \
                --deploy-mode ${SPARK_MODE} \
                --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
                --name orca-integrate-pod \
                --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata \
                --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata/ \
                --executor-cores 10 \
                --executor-memory 50g \
                --num-executors 4 \
                --total-executor-cores 40 \
                --driver-cores 10 \
                --driver-memory 50g \
                --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
                --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///bigdl/nfsdata/train.py,local:///bigdl/nfsdata/model.py \
                --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
                --conf spark.sql.catalogImplementation='in-memory' \
                --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
                --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
                local:///bigdl/nfsdata/train.py
                --cluster_mode spark-submit
                --remote_dir file:///bigdl/nfsdata/dataset
                "]
        securityContext:
          privileged: true
        env:
          - name: RUNTIME_K8S_SPARK_IMAGE
            value: intelanalytics/bigdl-spark-3.1.2:orca-2.1.0
          - name: RUNTIME_SPARK_MASTER
            value: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>
          - name: SPARK_MODE
            value: cluster
          - name: RUNTIME_K8S_SERVICE_ACCOUNT
            value: spark
          - name: BIGDL_HOME
            value: /opt/bigdl-2.1.0
          - name: SPARK_HOME
            value: /opt/spark
          - name: SPARK_VERSION
            value: 3.1.2
          - name: BIGDL_VERSION
            value: 2.1.0
        resources:
          requests:
            cpu: 1
          limits:
            cpu: 4
        volumeMounts:
          - name: nfs-storage
            mountPath: /bigdl/nfsdata
          - name: nfs-storage
            mountPath: /root/.kube/config
            subPath: kubeconfig
      volumes:
      - name: nfs-storage
        persistentVolumeClaim:
          claimName: nfsvolumeclaim
```

In the YAML file:
* `restartPolicy`: Restart policy for all Containers within the pod. One of Always, OnFailure, Never. Default to Always.
* `containers`: A single application Container that you want to run within a pod.
    * `name`: Name of the Container, each Container in a pod must have a unique name.
    * `image`: Name of the Container image.
    * `imagePullPolicy`: Image pull policy. One of Always, Never and IfNotPresent. Defaults to Always if `:latest` tag is specified, or IfNotPresent otherwise.
    * `command`: command for the containers that run in the Pod.
    * `args`: arguments to submit the spark application in the Pod. See more details of the `spark-submit` script in __[Section 6.2.2](#622-k8s-cluster)__.
    * `securityContext`: SecurityContext defines the security options the container should be run with.
    * `env`: List of environment variables to set in the Container, which will be used when submitting the application.
        * `env.name`: Name of the environment variable.
        * `env.value`: Value of the environment variable.
    * `resources`: Allocate resources in the cluster to each pod.
        * `resource.limits`: Limits describes the maximum amount of compute resources allowed.
        * `resource.requests`: Requests describes the minimum amount of compute resources required.
    * `volumeMounts`: Declare where to mount volumes into containers.
        * `name`: Match with the Name of a Volume.
        * `mountPath`: Path within the Container at which the volume should be mounted.
        * `subPath`: Path within the volume from which the Container's volume should be mounted.
    * `volume`: specify the volumes to provide for the Pod.
        * `persistentVolumeClaim`: mount a PersistentVolume into a Pod

Create a Pod and run Fashion-MNIST application based on the YAML file.
```bash
kubectl apply -f integrate_image_cluster.yaml
```

List all pods to find the driver pod (since the client pod only returns training status), which will be named as `orca-integrate-job-driver`.
```bash
# checkout training status
kubectl logs `orca-integrate-job-xxx`

# find out driver pod
kubectl get pods
```

View logs from the driver pod to retrive the training stats. 
```bash
# retrive training logs
kubectl logs `orca-integrate-job-driver`
```

After the task finish, you could delete the job as the command below.
```bash
kubectl delete job orca-integrate-job
```
