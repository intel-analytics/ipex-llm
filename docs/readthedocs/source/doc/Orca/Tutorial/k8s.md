# Run on Kubernetes Clusters

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Kubernetes (K8s) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/python/orca/tutorial/pytorch/FashionMNIST) as a working example.

The __Develop Node__ is the host machine where you launch the client container or create a Kubernetes Deployment. The **Client Container** is the created Docker container where you launch or submit your applications.

---
## 1. Basic Concepts
### 1.1 init_orca_context
A BigDL Orca program usually starts with the initialization of OrcaContext. For every BigDL Orca program, you should call `init_orca_context` at the beginning of the program as below:

```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode, master, container_image, 
                  cores, memory, num_nodes, driver_cores, driver_memory, 
                  extra_python_lib, penv_archive, conf)
```

In `init_orca_context`, you may specify necessary runtime configurations for running the example on K8s, including:
* `cluster_mode`: one of `"k8s-client"`, `"k8s-cluster"` or `"spark-submit"` when you run on K8s clusters.
* `master`: a URL format to specify the master address of the K8s cluster.
* `container_image`: a string that specifies the name of Docker container image for executors. The Docker container image for BigDL is `intelanalytics/bigdl-k8s`.
* `cores`: an integer that specifies the number of cores for each executor (default to be `2`).
* `memory`: a string that specifies the memory for each executor (default to be `"2g"`).
* `num_nodes`: an integer that specifies the number of executors (default to be `1`).
* `driver_cores`: an integer that specifies the number of cores for the driver node (default to be `4`).
* `driver_memory`: a string that specifies the memory for the driver node (default to be `"2g"`).
* `extra_python_lib`: a string that specifies the path to extra Python packages, separated by comma (default to be `None`). `.py`, `.zip` or `.egg` files are supported.
* `penv_archive`: a string that specifies the path to a packed conda archive (default to be `None`).
* `conf`: a dictionary to append extra conf for Spark (default to be `None`).

__Note__: 
* All arguments __except__ `cluster_mode` will be ignored when using [`spark-submit`](#use-spark-submit) or [`Kubernetes deployment`](#use-kubernetes-deployment) to submit and run Orca programs, in which case you are supposed to specify these configurations via the submit command or the YAML file.

After Orca programs finish, you should always call `stop_orca_context` at the end of the program to release resources and shutdown the underlying distributed runtime engine (such as Spark or Ray).
```python
from bigdl.orca import stop_orca_context

stop_orca_context()
```

For more details, please see [OrcaContext](../Overview/orca-context.md).


### 1.2 K8s-Client & K8s-Cluster
The difference between k8s-client mode and k8s-cluster mode is where you run your Spark driver. 

For k8s-client, the Spark driver runs in the client process (outside the K8s cluster), while for k8s-cluster the Spark driver runs inside the K8s cluster.

Please see more details in [K8s-Cluster](https://spark.apache.org/docs/latest/running-on-kubernetes.html#cluster-mode) and [K8s-Client](https://spark.apache.org/docs/latest/running-on-kubernetes.html#client-mode).

For **k8s-client** mode, you can directly find the driver logs in the console.

For **k8s-cluster** mode, a `driver-pod-name` (`train-py-fc5bec85fca28cb3-driver` in the following log) will be returned when the application is completed.
```
23-01-29 08:34:47 INFO  LoggingPodStatusWatcherImpl:57 - Application status for spark-9341aa0ec6b249ad974676c696398b4e (phase: Succeeded)
23-01-29 08:34:47 INFO  LoggingPodStatusWatcherImpl:57 - Container final statuses:
         container name: spark-kubernetes-driver
         container image: intelanalytics/bigdl-k8s:latest
         container state: terminated
         container started at: 2023-01-29T08:26:56Z
         container finished at: 2023-01-29T08:35:07Z
         exit code: 0
         termination reason: Completed
23-01-29 08:34:47 INFO  LoggingPodStatusWatcherImpl:57 - Application train.py with submission ID default:train-py-fc5bec85fca28cb3-driver finished
23-01-29 08:34:47 INFO  ShutdownHookManager:57 - Shutdown hook called
23-01-29 08:34:47 INFO  ShutdownHookManager:57 - Deleting directory /tmp/spark-fa8eeb45-bebf-4da9-9c0b-8bb59543842d
```

You can access the results of the driver pod on the __Develop Node__ following the commands below:

* Retrieve the logs on the driver pod:
```bash
kubectl logs <driver-pod-name>
```

* Check the pod status or get basic information of the driver pod:
```bash
kubectl describe pod <driver-pod-name>
```


### 1.3 Load Data from Volumes
When you are running programs on K8s, please load data from [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/) accessible to all K8s pods. We use Network File Systems (NFS) with path `/bigdl/nfsdata` in this tutorial as an example. You are recommended to put your working directory in the Volume (NFS) as well.

To load data from Volumes, please set the corresponding Volume configurations for spark using `--conf` option in Spark scripts or specifying `conf` in `init_orca_context`. Here we list the configurations for using NFS as Volume.

For **k8s-client** mode:
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of persistentVolumeClaim with volumnName `nfsvolumeclaim` to mount into executor pods.
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: specify the NFS path (`/bigdl/nfsdata` in our example) to be mounted as nfsvolumeclaim into executor pods.

Besides the above two configurations, you need to additionally set the following configurations for **k8s-cluster** mode:
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of persistentVolumeClaim with volumnName `nfsvolumeclaim` to mount into the driver pod.
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: specify the NFS path (`/bigdl/nfsdata` in our example) to be mounted as nfsvolumeclaim into the driver pod.
* `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for the driver pod.
* `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in k8s-cluster mode. In this example we use the NFS path as well.

Sample conf for NFS in the Fashion-MNIST example provided by this tutorial is as follows:
```python
{
    # For k8s-client mode
    "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
    "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
    
    # Additionally for k8s-cluster mode
    "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
    "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl/nfsdata",
    "spark.kubernetes.authenticate.driver.serviceAccountName": "spark",
    "spark.kubernetes.file.upload.path": "/bigdl/nfsdata/"
}
```

After mounting the Volume (NFS) into the BigDL container (see __[Section 2.2](#create-a-k8s-client-container)__ for more details), the Fashion-MNIST example could load data from NFS as local storage.

```python
import torch
import torchvision
import torchvision.transforms as transforms

def train_data_creator(config, batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.FashionMNIST(root="/bigdl/nfsdata/dataset", train=True, 
                                                 download=False, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)
    return trainloader
```

---
### 2 Pull Docker Image
Please pull the BigDL [`bigdl-k8s`](https://hub.docker.com/r/intelanalytics/bigdl-k8s/tags) image (built on top of Spark 3.1.3) from Docker Hub beforehand as follows:
```bash
# For the latest nightly build version
sudo docker pull intelanalytics/bigdl-k8s:latest

# For the release version, e.g. 2.2.0
sudo docker pull intelanalytics/bigdl-k8s:2.2.0
```

In the docker container:
- Spark is located at `/opt/spark`. Spark version is 3.1.3.
- BigDL is located at `/opt/bigdl-VERSION`. For the latest nightly build image, BigDL version would be `xxx-SNAPSHOT` (e.g. 2.3.0-SNAPSHOT).

---
## 3. Create BigDL K8s Container
Note that you can __skip__ this section if you want to run applications with [`Kubernetes deployment`](#use-kubernetes-deployment).

You need to create a BigDL K8s client container only when you use [`python` command](#use-python-command) or [`spark-submit`](#use-spark-submit).

### 3.1 Create a K8s Client Container
Please create the __Client Container__ using the script below:
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
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/bigdl-k8s:latest \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=nfsvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    intelanalytics/bigdl-k8s:latest bash
```

In the script:
* **Please switch the tag according to the BigDL image you pull.**
* **Please make sure you are mounting the correct Volume path (e.g. NFS) into the container.**
* `--net=host`: use the host network stack for the Docker container.
* `-v /etc/kubernetes:/etc/kubernetes`: specify the path of Kubernetes configurations to mount into the Docker container.
* `-v /root/.kube:/root/.kube`: specify the path of Kubernetes installation to mount into the Docker container.
* `-v /path/to/nfsdata:/bigdl/nfsdata`: mount NFS path on the host into the container as the specified path (e.g. "/bigdl/nfsdata").
* `NOTEBOOK_PORT`: an integer that specifies the port number for the Notebook. This is not necessary if you don't use notebook.
* `NOTEBOOK_TOKEN`: a string that specifies the token for Notebook. This is not necessary if you don't use notebook.
* `RUNTIME_SPARK_MASTER`: a URL format that specifies the Spark master: `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`.
* `RUNTIME_K8S_SERVICE_ACCOUNT`: a string that specifies the service account for the driver pod.
* `RUNTIME_K8S_SPARK_IMAGE`: the name of the BigDL K8s Docker image. Note that you need to change the version accordingly.
* `RUNTIME_PERSISTENT_VOLUME_CLAIM`: a string that specifies the Kubernetes volumeName (e.g. "nfsvolumeclaim").
* `RUNTIME_DRIVER_HOST`: a URL format that specifies the driver localhost (only required if you use k8s-client mode).

__Notes:__
* The __Client Container__ already contains all the required environment configurations for Spark and BigDL Orca.
* Spark executor containers are scheduled by K8s at runtime and you don't need to create them manually.


### 3.2 Launch the K8s Client Container
Once the container is created, a `containerID` would be returned and with which you can enter the container following the command below:
```bash
sudo docker exec -it <containerID> bash
```
In the remaining part of this tutorial, you are supposed to operate and run commands *__inside__* this __Client Container__.


---
## 4. Prepare Environment
In the launched BigDL K8s **Client Container** (if you use [`python` command](#use-python-command) or [`spark-submit`](#use-spark-submit)) or on the **Develop Node** (if you use [`Kubernetes deployment`](#use-kubernetes-deployment)), please setup the environment following the steps below:

- See [here](../Overview/install.md#install-anaconda) to install conda and prepare the Python environment.

- See [here](../Overview/install.md#to-install-orca-for-spark3) to install BigDL Orca in the created conda environment. *Note that if you use [`spark-submit`](#use-spark-submit) or [`Kubernetes deployment`](#use-kubernetes-deployment), please __skip__ this step and __DO NOT__ install BigDL Orca with pip install command in the conda environment.*

- You should install all the other Python libraries that you need in your program in the conda environment as well. `torch` and `torchvision` are needed to run the Fashion-MNIST example we provide:
```bash
pip install torch torchvision tqdm
```


---
## 5. Prepare Dataset
To run the Fashion-MNIST example provided by this tutorial on K8s, you should upload the dataset to a K8s Volume (e.g. NFS).

Please download the Fashion-MNIST dataset manually on your __Develop Node__ and put the data into the Volume. Note that PyTorch `FashionMNIST Dataset` requires unzipped files located in `FashionMNIST/raw/` under the dataset folder.

```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

# Copy the dataset files to the folder FashionMNIST/raw in NFS
cp /path/to/fashion-mnist/data/fashion/* /bigdl/nfsdata/dataset/FashionMNIST/raw

# Extract FashionMNIST archives
gzip -d /bigdl/nfsdata/dataset/FashionMNIST/raw/*
```

In the given example, you can specify the argument `--data_dir` to be the directory on NFS for the Fashion-MNIST dataset. The directory should contain `FashionMNIST/raw/train-images-idx3-ubyte` and `FashionMNIST/raw/t10k-images-idx3`.


---
## 6. Prepare Custom Modules
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) across the cluster by setting `--py-files` option in Spark scripts or specifying `extra_python_lib` in `init_orca_context`.

The FasionMNIST example needs to import the modules from [`model.py`](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/model.py).

__Note:__ Please upload the extra Python dependency files to the Volume (e.g. NFS) when running the program on k8s-cluster mode (see __[Section 6.2.2](#id2)__ for more details).

* When using [`python` command](#use-python-command), please specify `extra_python_lib` in `init_orca_context`.
```python
init_orca_context(..., extra_python_lib="/bigdl/nfsdata/model.py")
```
For more details, please see [BigDL Python Dependencies](https://bigdl.readthedocs.io/en/latest/doc/Orca/Overview/orca-context.html#python-dependencies).


* When using [`spark-submit`](#use-spark-submit), please specify `--py-files` option in the submit command.    
```bash
spark-submit
    ...
    --py-files file:///bigdl/nfsdata/model.py
    ...
```
For more details, please see [Spark Python Dependencies](https://spark.apache.org/docs/latest/submitting-applications.html). 


* After uploading `model.py` to K8s, you can import this custom module as follows:
```python
from model import model_creator, optimizer_creator
```

__Notes__:

If your program depends on a nested directory of Python files, you are recommended to follow the steps below to use a zipped package instead.

1. Compress the directory into a zipped package.
    ```bash
    zip -q -r FashionMNIST_zipped.zip FashionMNIST
    ```
2. Upload the zipped package (`FashionMNIST_zipped.zip`) to K8s by setting `--py-files` or specifying `extra_python_lib` as discussed above.

3. You can then import the custom modules from the unzipped file in your program as follows:
    ```python
    from FashionMNIST.model import model_creator, optimizer_creator
    ```


---
## 7. Run Jobs on K8s
In the following part, we will illustrate three ways to submit and run BigDL Orca applications on K8s.

* Use `python` command
* Use `spark-submit`
* Use Kubernetes Deployment

You can choose one of them based on your preference or cluster settings.

We provide the running command for the [Fashion-MNIST example](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/) in the __Client Container__ in this section.

### 7.1 Use `python` command
This is the easiest and most recommended way to run BigDL Orca on K8s as a normal Python program.

See [here](#init-orca-context) for the runtime configurations.

#### 7.1.1 K8s-Client
Run the example with the following command by setting the cluster_mode to "k8s-client":
```bash
python train.py --cluster_mode k8s-client --data_dir /bigdl/nfsdata/dataset
```


#### 7.1.2 K8s-Cluster
Before running the example on `k8s-cluster` mode in the __Client Container__, you should:

1. Pack the current activate conda environment to an archive:
    ```bash
    conda pack -o environment.tar.gz
    ```
2. Upload the conda archive to NFS:
    ```bash
    cp /path/to/environment.tar.gz /bigdl/nfsdata
    ```
3. Upload the Python script (`train.py` in our example) to NFS:
    ```bash
    cp /path/to/train.py /bigdl/nfsdata
    ```
4. Upload the extra Python dependency files (`model.py` in our example) to NFS:
    ```bash
    cp /path/to/model.py /bigdl/nfsdata
    ```

Run the example with the following command by setting the cluster_mode to "k8s-cluster":
```bash
python /bigdl/nfsdata/train.py --cluster_mode k8s-cluster --data_dir /bigdl/nfsdata/dataset
```


### 7.2 Use `spark-submit`

If you prefer to use `spark-submit`, please follow the steps below in the __Client Container__ before submitting the application. . 

1. Download the requirement file(s) from [here](https://github.com/intel-analytics/BigDL/tree/main/python/requirements/orca) and install the required Python libraries of BigDL Orca according to your needs.
    ```bash
    pip install -r /path/to/requirements.txt
    ```
    Note that you are recommended **NOT** to install BigDL Orca with pip install command in the conda environment if you use spark-submit to avoid possible conflicts.

2. Pack the current activate conda environment to an archive:
    ```bash
    conda pack -o environment.tar.gz
    ```

3. Set the cluster_mode to "spark-submit" in `init_orca_context`:
    ```python
    sc = init_orca_context(cluster_mode="spark-submit")
    ```

Some runtime configurations for Spark are as follows:

* `--master`: a URL format that specifies the Spark master: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>.
* `--name`: the name of the Spark application.
* `--conf spark.kubernetes.container.image`: the name of the BigDL K8s Docker image.
* `--num-executors`: the number of executors.
* `--executor-cores`: the number of cores for each executor.
* `--total-executor-cores`: the total number of executor cores.
* `--executor-memory`: the memory for each executor.
* `--driver-cores`: the number of cores for the driver.
* `--driver-memory`: the memory for the driver.
* `--properties-file`: the BigDL configuration properties to be uploaded to K8s.
* `--py-files`: the extra Python dependency files to be uploaded to K8s.
* `--archives`: the conda archive to be uploaded to K8s.
* `--conf spark.driver.extraClassPath`: upload and register BigDL jars files to the driver's classpath.
* `--conf spark.executor.extraClassPath`: upload and register BigDL jars files to the executors' classpath.
* `--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` to mount `persistentVolume` into executor pods.
* `--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: specify the path to be mounted as `persistentVolumeClaim` into executor pods.


#### 7.2.1 K8s Client
Submit and run the program for `k8s-client` mode following the `spark-submit` script below: 
```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode client \
    --name orca-k8s-client-tutorial \
    --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --num-executors 2 \
    --executor-cores 4 \
    --total-executor-cores 8 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives /path/to/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=python \
    --conf spark.pyspark.python=./environment/bin/python \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/path/to/model.py \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    train.py --cluster_mode spark-submit --data_dir /bigdl/nfsdata/dataset
```

In the `spark-submit` script:
* `deploy-mode`: set it to `client` when running programs on k8s-client mode.
* `--conf spark.driver.host`: the localhost for the driver pod.
* `--conf spark.pyspark.driver.python`: set the activate Python location in __Client Container__ as the driver's Python environment.
* `--conf spark.pyspark.python`: set the Python location in the conda archive as each executor's Python environment.


#### 7.2.2 K8s Cluster

Before running the example on `k8s-cluster` mode in the __Client Container__, you should:

1. Upload the conda archive to NFS:
    ```bash
    cp /path/to/environment.tar.gz /bigdl/nfsdata
    ```
2. Upload the Python script (`train.py` in our example) to NFS:
    ```bash
    cp /path/to/train.py /bigdl/nfsdata
    ```
3. Upload the extra Python dependency files (`model.py` in our example) to NFS:
    ```bash
    cp /path/to/model.py /bigdl/nfsdata
    ```

Submit and run the program for `k8s-cluster` mode following the `spark-submit` script below:
```bash
${SPARK_HOME}/bin/spark-submit \
    --master ${RUNTIME_SPARK_MASTER} \
    --deploy-mode cluster \
    --name orca-k8s-cluster-tutorial \
    --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
    --num-executors 2 \
    --executor-cores 4 \
    --total-executor-cores 8 \
    --executor-memory 2g \
    --driver-cores 2 \
    --driver-memory 2g \
    --archives file:///bigdl/nfsdata/environment.tar.gz#environment \
    --conf spark.pyspark.driver.python=environment/bin/python \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.kubernetes.file.upload.path=/bigdl/nfsdata \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,file:///bigdl/nfsdata/train.py,file:///bigdl/nfsdata/model.py \
    --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    file:///bigdl/nfsdata/train.py --cluster_mode spark-submit --data_dir /bigdl/nfsdata/dataset
```

In the `spark-submit` script:
* `deploy-mode`: set it to `cluster` when running programs on k8s-cluster mode.
* `--conf spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for the driver pod.
* `--conf spark.pyspark.driver.python`: set the Python location in the conda archive as the driver's Python environment.
* `--conf spark.pyspark.python`: also set the Python location in the conda archive as each executor's Python environment.
* `--conf spark.kubernetes.file.upload.path`: the path to store files at spark submit side in k8s-cluster mode.
* `--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` to mount `persistentVolume` into the driver pod.
* `--conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: specify the path to be mounted as `persistentVolumeClaim` into the driver pod.


### 7.3 Use Kubernetes Deployment
BigDL supports users who want to execute programs directly on __Develop Node__ to run an application by creating a Kubernetes Deployment object.
After preparing the [Conda environment](#prepare-environment) on the __Develop Node__, follow the steps below before submitting the application.

1. Download the requirement file(s) from [here](https://github.com/intel-analytics/BigDL/tree/main/python/requirements/orca) and install the required Python libraries of BigDL Orca according to your needs.
    ```bash
    pip install -r /path/to/requirements.txt
    ```
    Note that you are recommended **NOT** to install BigDL Orca with pip install command in the conda environment if you use spark-submit to avoid possible conflicts.

2. Pack the current activate conda environment to an archive before:
    ```bash
    conda pack -o environment.tar.gz
    ```

3. Upload the conda archive, Python script (`train.py` in our example) and extra Python dependency files (`model.py` in our example) to NFS.
    ```bash
    cp /path/to/environment.tar.gz /path/to/nfs

    cp /path/to/train.py /path/to/nfs

    cp /path/to/model.py /path/to/nfs
    ```

4. Set the cluster_mode to "spark-submit" in `init_orca_context`.
    ```python
    sc = init_orca_context(cluster_mode="spark-submit")
    ```

We define a Kubernetes Deployment in a YAML file. Some fields of the YAML are explained as follows:

* `metadata`: a nested object filed that every deployment object must specify.
    * `name`: a string that uniquely identifies this object and job. We use "orca-pytorch-job" in our example.
* `restartPolicy`: the restart policy for all containers within the pod. One of Always, OnFailure, Never. Default to Always.
* `containers`: a single application container to run within a pod.
    * `name`: the name of the container. Each container in a pod will have a unique name.
    * `image`: the name of the BigDL K8s Docker image. Note that you need to change the version accordingly.
    * `imagePullPolicy`: the pull policy of the docker image. One of Always, Never and IfNotPresent. Defaults to Always if `:latest` tag is specified, or IfNotPresent otherwise.
    * `command`: the command for the containers to run in the pod.
    * `args`: the arguments to submit the spark application in the pod. See more details in [`spark-submit`](#use-spark-submit).
    * `securityContext`: the security options the container should be run with.
    * `env`: a list of environment variables to set in the container, which will be used when submitting the application. Note that you need to change the environment variables including `BIGDL_VERSION` and `BIGDL_HOME` accordingly.
        * `name`: the name of the environment variable.
        * `value`: the value of the environment variable.
    * `resources`: the resources to be allocated in the cluster for each pod.
        * `requests`: the minimum amount of computation resources required.
        * `limits`: the maximum amount of computation resources allowed.
    * `volumeMounts`: the paths to mount Volumes into containers.
        * `name`: the name of a Volume.
        * `mountPath`: the path in the container to mount the Volume to.
        * `subPath`: the sub-path within the volume to mount into the container.
* `volumes`: specify the volumes for the pod. We use NFS as the persistentVolumeClaim in our example.


#### 7.3.1 K8s Client
BigDL has provided an example [orca-tutorial-k8s-client.yaml](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/docker/orca-tutorial-client.yaml)__ to directly run the Fashion-MNIST example for k8s-client mode.
Note that you need to change the configurations in the YAML file accordingly, including the version of the docker image, RUNTIME_SPARK_MASTER, BIGDL_VERSION and BIGDL_HOME.

You need to uncompress the conda archive in NFS before submitting the job:
```bash
cd /path/to/nfs
mkdir environment
tar -xzvf environment.tar.gz --directory environment
```

```bash
# orca-tutorial-k8s-client.yaml
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
        image: intelanalytics/bigdl-k8s:latest
        imagePullPolicy: IfNotPresent
        command: ["/bin/sh","-c"]
        args: ["
                export RUNTIME_DRIVER_HOST=$( hostname -I | awk '{print $1}' );
                ${SPARK_HOME}/bin/spark-submit \
                --master ${RUNTIME_SPARK_MASTER} \
                --deploy-mode ${SPARK_MODE} \
                --name orca-k8s-client-tutorial \
                --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
                --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName=nfsvolumeclaim \
                --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path=/bigdl/nfsdata/ \
                --conf spark.pyspark.driver.python=/bigdl/nfsdata/environment/bin/python \
                --conf spark.pyspark.python=/bigdl/nfsdata/environment/bin/python \
                --num-executors 2 \
                --executor-cores 4 \
                --executor-memory 2g \
                --total-executor-cores 8 \
                --driver-cores 2 \
                --driver-memory 2g \
                --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
                --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/bigdl/nfsdata/kai/model.py \
                --conf spark.kubernetes.executor.deleteOnTermination=True \
                --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
                --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
                /bigdl/nfsdata/train.py
                --cluster_mode spark-submit
                --data_dir /bigdl/nfsdata/dataset
                "]
        securityContext:
          privileged: true
        env:
          - name: RUNTIME_K8S_SPARK_IMAGE
            value: intelanalytics/bigdl-k8s:latest
          - name: RUNTIME_SPARK_MASTER
            value: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>
          - name: SPARK_MODE
            value: client
          - name: SPARK_VERSION
            value: 3.1.3
          - name: SPARK_HOME
            value: /opt/spark
          - name: BIGDL_VERSION
            value: 2.2.0-SNAPSHOT
          - name: BIGDL_HOME
            value: /opt/bigdl-2.2.0-SNAPSHOT
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


Submit the application using `kubectl`:
```bash
kubectl apply -f orca-tutorial-k8s-client.yaml
```

Note that you need to delete the job before re-submitting another one:
```bash
kubectl delete job orca-pytorch-job
```

After submitting the job, you can list all the pods and find the driver pod with name `orca-pytorch-job-xxx`:
```bash
kubectl get pods
kubectl get pods | grep orca-pytorch-job
```

Retrieve the logs on the driver pod:
```bash
kubectl logs orca-pytorch-job-xxx
```

After the task finishes, delete the job and all related pods if necessary:
```bash
kubectl delete job orca-pytorch-job
```

#### 7.3.2 K8s Cluster
BigDL has provided an example [orca-tutorial-k8s-cluster.yaml](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/docker/orca-tutorial-cluster.yaml)__ to run the Fashion-MNIST example for k8s-cluster mode.
Note that you need to change the configurations in the YAML file accordingly, including the version of the docker image, RUNTIME_SPARK_MASTER, BIGDL_VERSION and BIGDL_HOME.

```bash
orca-tutorial-k8s-cluster.yaml
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
                --data_dir file:///bigdl/nfsdata/dataset
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

Submit the application using `kubectl`:
```bash
kubectl apply -f orca-tutorial-k8s-cluster.yaml
```

Note that you need to delete the job before re-submitting another one:
```bash
kubectl delete job orca-pytorch-job
```

After submitting the job, you can list all the pods and find the driver pod with name `orca-pytorch-job-driver`.
```bash
kubectl get pods
kubectl get pods | grep orca-pytorch-job-driver
```

Retrieve the logs on the driver pod:
```bash
kubectl logs orca-pytorch-job-driver
```

After the task finishes, delete the job and all related pods if necessary:
```bash
kubectl delete job orca-pytorch-job
```
