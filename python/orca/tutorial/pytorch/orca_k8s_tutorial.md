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
* `cluster_mode`: a String that specifies the underlying cluster; valid value includes `"local"`, `"yarn-client"`, `"yarn-cluster"`, `"k8s-client"`, `"k8s-cluster"`, `"bigdl-submit"`, `"spark-submit"`, etc.
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
* All arguments __except__ `cluster_mode` will be ignored when using `spark-submit` to submit and run Orca programs.

After the Orca programs finished, you should call `stop_orca_context` at the end of the program to release resources and shutdown the underlying distributed runtime engine (such as Spark or Ray).
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
When you are running programs on K8s, please load data from a network file system (NFS) instead of the local file system.

After mounting the NFS into BigDL container (see __Section 2.2 & 4__), the Fashion-MNIST example could load data from NFS the same as loading from a local file system.

```python
import torch
import torchvision
import torchvision.transforms as transforms
from bigdl.orca.data.file import get_remote_file_to_local

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root="/bigdl/nfsdata/dataset", train=True, 
                                                 download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    return trainloader
```

# 2. Create & Lunch BigDL K8s Container 
## 2.1 Pull Docker Image
Please pull the prebuilt BigDL `bigdl-k8s` image from [Docker Hub](https://hub.docker.com/r/intelanalytics/bigdl-k8s/tags) as follows:
```bash
sudo docker pull intelanalytics/bigdl-k8s:latest
```

__Note:__
* The latest BigDL image is built on top of Spark 3.1.2.

## 2.2 Create a K8s Client Container
Please launch the __Client Container__ following the script below:
```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    intelanalytics/bigdl-k8s:latest bash
```

In the script:
* `--net=host`: use the host network stack for the Docker container;
* `-v /etc/kubernetes:/etc/kubernetes`: specify the path of kubernetes configurations;
* `-v /root/.kube:/root/.kube`: specify the path of kubernetes installation;

__Note:__
* The __Client Container__ contains all the required environment and libraries except Hadoop/K8s configs.
* You needn't to create an __Executor Container__ manually, which is scheduled by K8s at runtime.


We __highly recommand__ you to specify more arguments when using `spark-submit` script:
```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v /path/to/nfsdata:/bigdl/data \
    -e NOTEBOOK_PORT=12345 \
    -e NOTEBOOK_TOKEN="your-token" \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=spark \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/bigdl-k8s:latest \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=nfsvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=2 \
    -e RUNTIME_EXECUTOR_CORES=2 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    intelanalytics/bigdl-k8s:latest bash 
```

In the script:
* `--net=host`: use the host network stack for the Docker container;
* `/etc/kubernetes:/etc/kubernetes`: specify the path of kubernetes configurations;
* `/root/.kube:/root/.kube`: specify the path of kubernetes installation;
* `/path/to/nfsdata:/bigdl/data`: mount NFS path on host in container as the sepcified path in value; 
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
Once the container is created, docker image would return the <containerID>.

Please launch the container following the command below:
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
# Please change Python version to 3.6 if you need a Python 3.6 environment.
conda create -n env python=3.7 
conda activate env
```

Alternatively, You can install the latest release version of BigDL Orca (built on top of Spark 3.1.2) as follows:
```bash
pip install bigdl-spark3
```

You can install the latest nightly build of BigDL as follows:
```bash
pip install --pre --upgrade bigdl-spark3
```

__Note:__
* Using Conda to install BigDL will automatically install libraries including `conda-pack`, `torch`, `torchmetrics`, `torchvision`, `pandas`, `numpy`, `pyspark==3.1.2`, and etc.
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

* You also need to install any additional python libraries that your application depends on in this Conda environment.

* It's only need for you to install needed Python libraries using Conda, since the BigDL K8s container has already setup `JAVA_HOME`, `BIGDL_HOME`, `SPARK_HOME`, `SPARK_VERSION`, etc.

Please see more details in [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).


# 4. Prepare Dataset
To run the example provided by this tutorial on K8s, you should upload the dataset to a network file system (NFS).

Please download the Fashion-MNIST dataset manually on your __Host Node__ (where you launch docker image). 

```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

mv /path/to/fashion-mnist/data/fashion /bigdl/nfsdata/dataset
```

# 5. Prepare Custom Modules
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) to the executors by setting `--py-files` option in Spark scripts or `extra_python_lib` in `init_orca_context`. 

The FasionMNIST example needs to import modules from `model.py`.
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
                      master=os.environ.get("RUNTIME_SPARK_MASTER"),
                      container_image="intelanalytics/bigdl-k8s:latest",
                      extra_python_lib="model.py", conf=conf)
    ```

    __Note:__
    * Please upload the extra Python dependency files to NFS and use it from NFS instead (see __Section 6.1.2__).
        ```python
        init_orca_context(cluster_mode, num_nodes, cores, memory, master,
                          container_image, conf, extra_python_lib="file:///bigdl/nfsdata/model.py")
        ```


    Please see more details in [Orca Document](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html#python-dependencies).

* When using `spark-submit` script, please specify `--py-files` option in the script.
    
    ```bash
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,model.py
    ```

    __Note:__
    * Please upload the extra Python dependency files to NFS and use it from NFS instead (see __Section 6.2.2__).
        ```bash
        --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,file:///bigdl/nfsdata/model.py \
        ```

    Then import custom modules:
    ```python
    from bigdl.orca import init_orca_context, stop_orca_context
    from model import model_creator, optimizer_creator

    init_orca_context(cluster_mode="spark-submit")
    ```

    Please see more details in [Spark Document](https://spark.apache.org/docs/latest/submitting-applications.html). 

__Note:__
* You could follow the steps below to use a zipped package instead ( recommended if your program depends on a nested directory of Python files) :
    1. Compress the directory into a Zipped Package.
        ```bash
        zip -q -r FashionMNIST_zipped.zip FashionMNIST
        ```
    2. Please follow the same method as before (using `.py` files) to upload the zipped package (`FashionMNIST_zipped.zip`) to K8s. 
    3. You should import custom modules from the unzipped file as below.
        ```python
        from FashionMNIST.model import model_creator, optimizer_creator
        ```


# 6. Run Jobs on K8s
In the following part, we will show you how to submit and run the Orca example on K8s:
* Use `python` command
* Use `spark-submit` script

## 6.1 Use `python` command
### 6.1.1 K8s-Client
Before running the example on `k8s-client` mode, you should:
On the __Client Container__:
1. Please call `init_orca_context` at very begining part of each Orca program.
    ```python
    from bigdl.orca import init_orca_context, stop_orca_context

    conf={
          "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
          "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
        }

    init_orca_context(cluster_mode="k8s-client", num_nodes=2, cores=2, memory="2g",
                      master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>",
                      container_image="intelanalytics/bigdl-k8s:latest",
                      extra_python_lib="model.py", conf=conf)
    ```

    To load data from NFS, please use the following configuration propeties: 
    * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into executor pods;
    * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;

2. Using Conda to install BigDL and needed Python dependency libraries (see __Section 3__).

Run the example:
```bash
python train.py --cluster_mode k8s-client --remote_dir /bigdl/data
```

In the script:
* `cluster_mode`: set the cluster_mode in `init_orca_context`.
* `remote_dir`: directory on NFS for the dataset (see __Section 4__) and saving the model.


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
                          container_image="intelanalytics/bigdl-k8s:latest",
                          penv_archive="file:///bigdl2.0/data/sgwhat/environment.tar.gz",
                          extra_python_lib="/bigdl/nfsdata/model.py", conf=conf)
        ```

        When running Orca programs on `k8s-cluster` mode, please use the following additional configuration propeties: 
        * `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into driver pod;
        * `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to driver pod on the NFS path specified in value;
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount `persistentVolume` into executor pods;
        * `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: add volumeName `nfsvolumeclaim` of the volumeType `persistentVolumeClaim` to executor pods on the NFS path specified in value;
        * `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for driver pod;
        * `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in cluster mode;
    
    2. Using Conda to install BigDL and needed Python dependency libraries (see __Section 3__).

    3. Pack the current activate Conda environment to an archive.
        ```
        conda pack -o environment.tar.gz
        ```

* On the __Host Node__:
    1. Upload the Conda archive to NFS.
        ```bash
        docker cp <containerID>:/opt/spark/work-dir/environment.tar.gz /bigdl/nfsdata
        ```
    2. Upload the example Python file to NFS.
        ```bash
        docker cp <containerID>:/opt/spark/work-dir/train.py /bigdl/nfsdata
        ```
    3. Upload the extra Python dependency file to NFS.
        ```bash
        docker cp <containerID>:/opt/spark/work-dir/model.py /bigdl/nfsdata
        ```

Run the example on __Client Container__:
```bash
python /bigdl/nfsdata/train.py --cluster_mode k8s-cluster --remote_dir /bigdl/nfsdata
```

In the script:
* `cluster_mode`: set the cluster_mode in `init_orca_context`.
* `remote_dir`: directory on NFS for the dataset (see __Section 4__) and saving the model.

__Note:__
* It will return a <driver-pod-name> when the application is completed.

Please retreive training stats on the __Host Node__ following the commands below:
* Retrive training logs on the driver pod:
    ```bash
    kubectl logs <spark-driver-pod>
    ```

* Check pod status or get basic informations around pod:
    ```bash
    kubectl describe pod <spark-driver-pod>
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
    2. Using Conda to install BigDL and needed Python dependency libraries (see __Section 3__).
    3. Pack the current activate Conda environment to an archive.
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
    --py-files ${BIGDL_HOME}/python/bigdl-spark_3.1.2-2.1.0-SNAPSHOT-python-api.zip,/path/to/train.py,/path/to/model.py \
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
* `remote_dir`: directory on NFS for the dataset (see __Section 4__) and saving the model.


### 6.2.2 K8s Cluster
Before submitting the example on `k8s-cluster` mode, you should:
* On the __Client Container__:
    1. Please call `init_orca_context` at very begining part of each Orca program.
        ```python
        from bigdl.orca import init_orca_context

        init_orca_context(cluster_mode="spark-submit")
        ```
    2. Using Conda to install BigDL and needed Python dependency libraries (see __Section 3__).
    3. Pack the current activate Conda environment to an archive.
        ```bash
        conda pack -o environment.tar.gz
        ```

* On the __Host Node__ (where you launch the __Client Container__):
    1. Upload Conda archive to NFS.
        ```bash
        docker cp <containerID>:/path/to/environment.tar.gz /bigdl/nfsdata
        ```

    2. Upload the example python files to NFS.
        ```bash
        docker cp <containerID>:/path/to/train.py /bigdl/nfsdata
        ```

    3. Upload the extra Python dependencies to NFS.
        ```bash
        docker cp <containerID>:/path/to/model.py /bigdl/nfsdata
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
* `remote_dir`: directory on NFS for the dataset (see __Section 4__) and saving the model.


Please retrieve training stats on the __Host Node__ following the commands below:
* Retrive training logs on the driver pod:
    ```bash
    kubectl logs <spark-driver-pod>
    ```

* Check pod status or get basic informations around pod using:
    ```bash
    kubectl describe pod <spark-driver-pod>
    ```
