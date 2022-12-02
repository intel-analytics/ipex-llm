# Run on Kubernetes Clusters

This tutorial provides a step-by-step guide on how to run BigDL-Orca programs on Kubernetes (K8s) clusters, using a [PyTorch Fashin-MNIST program](https://github.com/intel-analytics/BigDL/tree/main/python/orca/tutorial/pytorch/FashionMNIST) as a working example.

The **Client Container** that appears in this tutorial refer to the docker container where you launch or submit your applications. The __Develop Node__ is the host machine where you launch the client container.

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
* `container_image`: a string that specifies the name of docker container image for executors.
* `cores`: an integer that specifies the number of cores for each executor (default to be `2`).
* `memory`: a string that specifies the memory for each executor (default to be `"2g"`).
* `num_nodes`: an integer that specifies the number of executors (default to be `1`).
* `driver_cores`: an integer that specifies the number of cores for the driver node (default to be `4`).
* `driver_memory`: a string that specifies the memory for the driver node (default to be `"1g"`).
* `extra_python_lib`: a string that specifies the path to extra Python packages, separated by comma (default to be `None`). `.py`, `.zip` or `.egg` files are supported.
* `penv_archive`: a string that specifies the path to a packed conda archive (default to be `None`).
* `conf`: a dictionary to append extra conf for Spark (default to be `None`).

__Note__: 
* All arguments __except__ `cluster_mode` will be ignored when using [`spark-submit`](#use-spark-submit) or [`Kubernetes deployment`](#use-kubernetes-deployment-with-conda-archive) to submit and run Orca programs, in which case you are supposed to specify these configurations via the submit command or the YAML file.

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

For **k8s-cluster** mode, a `driver pod name` will be returned when the application is completed. You can retrieve the results on the __Develop Node__ following the commands below:

* Retrieve the logs on the driver pod:
```bash
kubectl logs <driver-pod-name>
```

* Check the pod status or get basic information of the driver pod:
```bash
kubectl describe pod <driver-pod-name>
```


### 1.3 Load Data from Volumes
When you are running programs on K8s, please load data from [Volumes](https://kubernetes.io/docs/concepts/storage/volumes/) accessible to all K8s pods. We use Network File Systems (NFS) with path `/bigdl/nfsdata` in this tutorial as an example.

To load data from Volumes, please set the corresponding Volume configurations for spark using `--conf` option in Spark scripts or specifying `conf` in `init_orca_context`. Here we list the configurations for using NFS as Volume.

For **k8s-client** mode:
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount into executor pods.
* `spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: specify the NFS path to be mounted as `nfsvolumeclaim` to executor pods.

Besides the above two configurations, you need to additionally set the following configurations for **k8s-cluster** mode:
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName`: specify the claim name of `persistentVolumeClaim` with volumnName `nfsvolumeclaim` to mount into the driver pod.
* `spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path`: specify the NFS path to be mounted as `nfsvolumeclaim` to the driver pod.
* `spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for the driver pod.
* `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in k8s-cluster mode.

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
## 2. Create BigDL K8s Container 
### 2.1 Pull Docker Image
Please pull the BigDL [`bigdl-k8s`]((https://hub.docker.com/r/intelanalytics/bigdl-k8s/tags)) image (built on top of Spark 3.1.3) from Docker Hub as follows:
```bash
# For the latest nightly build version
sudo docker pull intelanalytics/bigdl-k8s:latest

# For the release version, e.g. 2.1.0
sudo docker pull intelanalytics/bigdl-k8s:2.1.0
```


### 2.2 Create a K8s Client Container
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
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=2 \
    -e RUNTIME_EXECUTOR_CORES=4 \
    -e RUNTIME_EXECUTOR_MEMORY=2g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=8 \
    -e RUNTIME_DRIVER_CORES=2 \
    -e RUNTIME_DRIVER_MEMORY=2g \
    intelanalytics/bigdl-k8s:latest bash
```

In the script:
* **Please switch the tag according to the BigDL image you pull.**
* **Please make sure you are mounting the correct Volume path (e.g. NFS) into the container.**
* `--net=host`: use the host network stack for the Docker container.
* `-v /etc/kubernetes:/etc/kubernetes`: specify the path of Kubernetes configurations to mount into the Docker container.
* `-v /root/.kube:/root/.kube`: specify the path of Kubernetes installation to mount into the Docker container.
* `-v /path/to/nfsdata:/bigdl/nfsdata`: mount NFS path on the host into the container as the specified path (e.g. "/bigdl/nfsdata").
* `NOTEBOOK_PORT`: an integer that specifies the port number for the Notebook (only required if you use notebook).
* `NOTEBOOK_TOKEN`: a string that specifies the token for Notebook (only required if you use notebook).
* `RUNTIME_SPARK_MASTER`: a URL format that specifies the Spark master: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>.
* `RUNTIME_K8S_SERVICE_ACCOUNT`: a string that specifies the service account for driver pod.
* `RUNTIME_K8S_SPARK_IMAGE`: the name of the BigDL K8s docker image.
* `RUNTIME_PERSISTENT_VOLUME_CLAIM`: a string that specifies the Kubernetes volumeName (e.g. "nfsvolumeclaim").
* `RUNTIME_DRIVER_HOST`: a URL format that specifies the driver localhost (only required by k8s-client mode).
* `RUNTIME_DRIVER_PORT`: a string that specifies the driver port (only required by k8s-client mode).
* `RUNTIME_EXECUTOR_INSTANCES`: an integer that specifies the number of executors.
* `RUNTIME_EXECUTOR_CORES`: an integer that specifies the number of cores for each executor.
* `RUNTIME_EXECUTOR_MEMORY`: a string that specifies the memory for each executor.
* `RUNTIME_TOTAL_EXECUTOR_CORES`: an integer that specifies the number of cores for all executors.
* `RUNTIME_DRIVER_CORES`: an integer that specifies the number of cores for the driver node.
* `RUNTIME_DRIVER_MEMORY`: a string that specifies the memory for the driver node.

__Notes:__
* The __Client Container__ contains all the required environment except K8s configurations.
* You don't need to create Spark executor containers manually, which are scheduled by K8s at runtime.


### 2.3 Launch the K8s Client Container
Once the container is created, a `containerID` would be returned and with which you can enter the container following the command below:
```bash
sudo docker exec -it <containerID> bash
```
In the remaining part of this tutorial, you are supposed to operate and run commands inside this __Client Container__.


---
## 3. Prepare Environment
In the launched BigDL K8s **Client Container**, please setup the environment following the steps below:

- See [here](../Overview/install.md#install-anaconda) to install conda and prepare the Python environment.

- See [here](../Overview/install.md#to-install-orca-for-spark3) to install BigDL Orca in the created conda environment.

- You should install all the other Python libraries that you need in your program in the conda environment as well. `torch` and `torchvision` are needed to run the Fashion-MNIST example:
```bash
pip install torch torchvision
```

- For more details, please see [Python User Guide](https://bigdl.readthedocs.io/en/latest/doc/UserGuide/python.html).


---
## 4. Prepare Dataset
To run the Fashion-MNIST example provided by this tutorial on K8s, you should upload the dataset to a K8s Volume (e.g. NFS).

Please download the Fashion-MNIST dataset manually on your __Develop Node__ and put the data into the Volume. Note that PyTorch `FashionMNIST Dataset` requires unzipped files located in `FashionMNIST/raw/` under the root folder.

```bash
# PyTorch official dataset download link
git clone https://github.com/zalandoresearch/fashion-mnist.git

# Move the dataset to NFS under the folder FashionMNIST/raw
mv /path/to/fashion-mnist/data/fashion /bigdl/nfsdata/dataset/FashionMNIST/raw

# Extract FashionMNIST archives
gzip -dk /bigdl/nfsdata/dataset/FashionMNIST/raw/*
```

In the given example, you can specify the argument `--remote_dir` to be the directory on NFS for the Fashion-MNIST dataset.


---
## 5. Prepare Custom Modules
Spark allows to upload Python files(`.py`), and zipped Python packages(`.zip`) across the cluster by setting `--py-files` option in Spark scripts or specifying `extra_python_lib` in `init_orca_context`.

The FasionMNIST example needs to import modules from `model.py`.

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

__Note__:

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
## 6. Run Jobs on K8s
In the following part, we will illustrate four ways to submit and run BigDL Orca applications on K8s.

* Use `python` command
* Use `spark-submit`
* Use Kubernetes Deployment (with Conda Archive)
* Use Kubernetes Deployment (with Integrated Image)

You can choose one of them based on your preference or cluster settings.

We provide the running command for the [Fashion-MNIST example](https://github.com/intel-analytics/BigDL/blob/main/python/orca/tutorial/pytorch/FashionMNIST/) in the __Client Container__ in this section.

### 6.1 Use `python` command
This is the easiest and most recommended way to run BigDL Orca on K8s as a normal Python program.

See [here](#init-orca-context) for the runtime configurations.

#### 6.1.1 K8s-Client
Run the example with the following command by setting the cluster_mode to "k8s-client":
```bash
python train.py --cluster_mode k8s-client --remote_dir file:///bigdl/nfsdata/dataset
```


#### 6.1.2 K8s-Cluster
Before running the example on `k8s-cluster` mode, you should:
* In the __Client Container__:

Pack the current activate conda environment to an archive:
```bash
conda pack -o environment.tar.gz
```

* On the __Develop Node__:
1. Upload the conda archive to NFS.
```bash
docker cp <containerID>:/path/to/environment.tar.gz /bigdl/nfsdata
```
2. Upload the example Python file to NFS.
```bash
cp /path/to/train.py /bigdl/nfsdata
```
3. Upload the extra Python dependency files to NFS.
```bash
cp /path/to/model.py /bigdl/nfsdata
```

Run the example with the following command by setting the cluster_mode to “k8s-cluster”:
```bash
python /bigdl/nfsdata/train.py --cluster_mode k8s-cluster --remote_dir /bigdl/nfsdata/dataset
```


### 6.2 Use `spark-submit`

Set the cluster_mode to "bigdl-submit" in `init_orca_context`.
```python
init_orca_context(cluster_mode="spark-submit")
```

Pack the current activate conda environment to an archive in the __Client Container__:
```bash
conda pack -o environment.tar.gz
```

Some runtime configurations for Spark are as follows:

* `--master`: a URL format that specifies the Spark master: k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>.
* `--name`: the name of the Spark application.
* `--conf spark.kubernetes.container.image`: the name of the BigDL K8s docker image.
* `--conf spark.kubernetes.authenticate.driver.serviceAccountName`: the service account for the driver pod.
* `--conf spark.executor.instances`: the number of executors.
* `--executor-memory`: the memory for each executor.
* `--driver-memory`: the memory for the driver node.
* `--executor-cores`: the number of cores for each executor.
* `--total-executor-cores`: the total number of executor cores.
* `--properties-file`: the BigDL configuration properties to be uploaded to K8s.
* `--py-files`: the extra Python dependency files to be uploaded to K8s.
* `--archives`: the conda archive to be uploaded to K8s.
* `--conf spark.driver.extraClassPath`: upload and register BigDL jars files to the driver's classpath.
* `--conf spark.executor.extraClassPath`: upload and register BigDL jars files to the executors' classpath.
* `--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName`: specify the claim name of `persistentVolumeClaim` to mount `persistentVolume` into executor pods.
* `--conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path`: specify the path to be mounted as `persistentVolumeClaim` to executor pods.


#### 6.2.1 K8s Client
Submit and run the program for `k8s-client` mode following the `spark-submit` script below: 
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
    --conf spark.pyspark.python=./environment/bin/python \
    --archives /path/to/environment.tar.gz#environment \
    --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
    --py-files ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,/path/to/train.py,/path/to/model.py \
    --conf spark.driver.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.executor.extraClassPath=${BIGDL_HOME}/jars/* \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
    --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/bigdl/nfsdata \
    train.py --cluster_mode spark-submit --remote_dir /bigdl/nfsdata/dataset
```

In the `spark-submit` script:
* `deploy-mode`: set it to `client` when running programs on k8s-client mode.
* `--conf spark.driver.host`: the localhost for the driver pod.
* `--conf spark.pyspark.driver.python`: set the activate Python location in __Client Container__ as the driver's Python environment.
* `--conf spark.pyspark.python`: set the Python location in conda archive as each executor's Python environment.


#### 6.2.2 K8s Cluster

* On the __Develop Node__:
1. Upload the conda archive to NFS.
```bash
docker cp <containerID>:/path/to/environment.tar.gz /bigdl/nfsdata
```
2. Upload the example Python file to NFS.
```bash
cp /path/to/train.py /bigdl/nfsdata
```
3. Upload the extra Python dependency files to NFS.
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
    --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
    --archives file:///bigdl/nfsdata/environment.tar.gz#environment \
    --conf spark.pyspark.python=environment/bin/python \
    --conf spark.executorEnv.PYTHONHOME=environment \
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
    file:///bigdl/nfsdata/train.py --cluster_mode spark-submit --remote_dir /bigdl/nfsdata/dataset
```

In the `spark-submit` script:
* `deploy-mode`: set it to `cluster` when running programs on k8s-cluster mode.
* `spark.pyspark.python`: sset the Python location in conda archive as each executor's Python environment.
* `spark.executorEnv.PYTHONHOME`: the search path of Python libraries on executor pods.
* `spark.kubernetes.file.upload.path`: the path to store files at spark submit side in k8s-cluster mode.


### 6.3 Use Kubernetes Deployment (with Conda Archive)
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

#### 6.3.1 K8s Client
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

#### 6.3.2 K8s Cluster
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


### 6.4 Use Kubernetes Deployment (without Integrated Image)
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

#### 6.4.1 K8s Client
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

#### 6.4.2 K8s Cluster
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
