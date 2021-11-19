# K8s User Guide

---

### **1. Pull `bigdl-k8s` Docker Image**

You may pull the prebuilt  BigDL `bigdl-k8s` Image from [Docker Hub](https://hub.docker.com/r/intelanalytics/bigdl-k8s/tags) as follows:

```bash
sudo docker pull intelanalytics/bigdl-k8s:latest
```

**Speed up pulling image by adding mirrors**

To speed up pulling the image from DockerHub, you may add the registry-mirrors key and value by editing `daemon.json` (located in `/etc/docker/` folder on Linux):
```
{
  "registry-mirrors": ["https://<my-docker-mirror-host>"]
}
```
For instance, users in China may add the USTC mirror as follows:
```
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
```

After that, flush changes and restart docker：

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### **2. Launch a Client Container**

You can submit BigDL application from a client container that provides the required environment.

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    intelanalytics/hyper-zoo:latest bash
```

**Note:** to create the client container, `-v /etc/kubernetes:/etc/kubernetes:` and `-v /root/.kube:/root/.kube` are required to specify the path of kube config and installation.

You can specify more arguments:

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -e NOTEBOOK_PORT=12345 \
    -e NOTEBOOK_TOKEN="your-token" \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=account \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/bigdl-k8s:latest \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=myvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=4 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    intelanalytics/bigdl-k8s:latest bash 
```

- NOTEBOOK_PORT value 12345 is a user specified port number.
- NOTEBOOK_TOKEN value "your-token" is a user specified string.
- http_proxy/https_proxy is to specify http proxy/https_proxy.
- RUNTIME_SPARK_MASTER is to specify spark master, which should be `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>` or `spark://<spark-master-host>:<spark-master-port>`. 
- RUNTIME_K8S_SERVICE_ACCOUNT is service account for driver pod. Please refer to k8s [RBAC](https://spark.apache.org/docs/latest/running-on-kubernetes.html#rbac).
- RUNTIME_K8S_SPARK_IMAGE is the k8s image.
- RUNTIME_PERSISTENT_VOLUME_CLAIM is to specify [Kubernetes volume](https://spark.apache.org/docs/latest/running-on-kubernetes.html#volume-mounts) mount. We are supposed to use volume mount to store or receive data.
- RUNTIME_DRIVER_HOST/RUNTIME_DRIVER_PORT is to specify driver localhost and port number (only required when submitting jobs via kubernetes client mode).
- Other environment variables are for spark configuration setting. The default values in this image are listed above. Replace the values as you need.

Once the container is created, execute the container:

```bash
sudo docker exec -it <containerID> bash
```

You will login into the container and see this as the output:

```
root@[hostname]:/opt/spark/work-dir# 
```

`/opt/spark/work-dir` is the spark work path. 

The `/opt` directory contains:

- download-bigdl.sh is used for downloading BigDL distributions.
- start-notebook-spark.sh is used for starting the jupyter notebook on standard spark cluster. 
- start-notebook-k8s.sh is used for starting the jupyter notebook on k8s cluster.
- bigdl-x.x-SNAPSHOT is `BIGDL_HOME`, which is the home of BigDL distribution.
- bigdl-examples directory contains downloaded python example code.
- install-conda-env.sh is displayed that conda env and python dependencies are installed.
- jdk is the jdk home.
- spark is the spark home.
- redis is the redis home.

### **3. Run BigDL Examples on k8s**

_**Note**: Please make sure `kubectl` has appropriate permission to create, list and delete pod._

_**Note**: Please refer to section 4 for some know issues._

#### **3.1 K8s client mode**

We recommend using `init_orca_context` at the very beginning of your code (e.g. in script.py) to initiate and run BigDL on standard K8s clusters in [client mode](http://spark.apache.org/docs/latest/running-on-kubernetes.html#client-mode).

```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="k8s", master="k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>",
                  container_image="intelanalytics/bigdl-k8s:latest",
                  num_nodes=2, cores=2,
                  conf={"spark.driver.host": "x.x.x.x",
                        "spark.driver.port": "x"
                       })
```

Execute `python script.py` to run your program on k8s cluster directly.

#### **3.2 K8s cluster mode**

For k8s [cluster mode](https://spark.apache.org/docs/3.1.2/running-on-kubernetes.html#cluster-mode), you can call `init_orca_context` and specify cluster_mode to be "spark-submit" in your python script (e.g. in script.py):

```python
from bigdl.orca import init_orca_context

init_orca_context(cluster_mode="spark-submit")
```

Use spark-submit to submit your BigDL program:

```bash
${SPARK_HOME}/bin/spark-submit \
  --master k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=account \
  --name bigdl \
  --conf spark.kubernetes.container.image="intelanalytics/bigdl-k8s:latest" \
  --conf spark.kubernetes.container.image.pullPolicy=Always \
  --conf spark.pyspark.driver.python=./environment/bin/python \
  --conf spark.pyspark.python=./environment/bin/python \
  --conf spark.executor.instances=1 \
  --executor-memory 10g \
  --driver-memory 10g \
  --executor-cores 8 \
  --num-executors 2 \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --py-files local://${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip,local:///path/script.py
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/* \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/* \
  local:///path/script.py
```

#### **3.3 Run Jupyter Notebooks**

After a Docker container is launched and user login into the container, you can start the Jupyter Notebook service inside the container.

In the `/opt` directory, run this command line to start the Jupyter Notebook service:
```
./start-notebook-k8s.sh
```

You will see the output message like below. This means the Jupyter Notebook service has started successfully within the container.
```
[I 23:51:08.456 NotebookApp] Serving notebooks from local directory: /opt/analytics-zoo-0.11.0-SNAPSHOT/apps
[I 23:51:08.456 NotebookApp] Jupyter Notebook 6.2.0 is running at:
[I 23:51:08.456 NotebookApp] http://xxxx:12345/?token=...
[I 23:51:08.457 NotebookApp]  or http://127.0.0.1:12345/?token=...
[I 23:51:08.457 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Then, refer [docker guide](./docker.md) to open Jupyter Notebook service from a browser and run notebook.

#### **3.4 Run Scala programs**

Use spark-submit to submit your BigDL program.  e.g., run [anomalydetection](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/anomalydetection) example (running in either local mode or cluster mode) as follows:

```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode client \
  --conf spark.driver.host=${RUNTIME_DRIVER_HOST} \
  --conf spark.driver.port=${RUNTIME_DRIVER_PORT} \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name bigdl \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/path \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/path \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${BIGDL_HOME}/conf/spark-bigdl.conf \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=local://${BIGDL_HOME}/jars/*  \
  --conf spark.executor.extraClassPath=local://${BIGDL_HOME}/jars/*  \
  --class com.intel.analytics.bigdl.dllib.examples.nnframes.imageInference.ImageTransferLearning \
  ${BIGDL_HOME}/python/bigdl-spark_${SPARK_VERSION}-${BIGDL_VERSION}-python-api.zip \
  --inputDir /path
```

Options:

- --master: the spark mater, must be a URL with the format `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`. 
- --deploy-mode: submit application in client/cluster mode.
- --name: the Spark application name.
- --conf: to specify k8s service account, container image to use for the Spark application, driver volumes name and path, label of pods, spark driver and executor configuration, etc. You can refer to [spark configuration](https://spark.apache.org/docs/latest/configuration.html) and [spark on k8s configuration](https://spark.apache.org/docs/latest/running-on-kubernetes.html#configuration) for more details.
- --properties-file: the customized conf properties.
- --py-files: the extra python packages is needed.
- --class: scala example class name.
- --inputDir: input data path of the nnframe example. The data path is the mounted filesystem of the host. Refer to more details by [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes).

### **4 Know issues**

This section shows some common topics for both client mode and cluster mode.

#### **4.1 How to specify python environment**

The k8s image provides two conda python environments, "pytf1" and "pytf2" are mainly distinguished with tensorflow version. "pytf1" is installed in "/usr/local/envs/pytf1/bin/python". "pytf2" is installed in "/usr/local/envs/pytf2/bin/python".
Specify on both the driver and executor in client mode:
```python
init_orca_context(..., conf={"spark.pyspark.driver.python": "/usr/local/envs/pytf1/bin/python",
                             "spark.pyspark.python": "/usr/local/envs/pytf1/bin/python",
                            })
```
Specify on both the driver and executor in cluster mode:
```bash
${SPARK_HOME}/bin/spark-submit \
  --... ...\
  --conf spark.pyspark.driver.python=/usr/local/envs/pytf1/bin/python \
  --conf spark.pyspark.python=/usr/local/envs/pytf1/bin/python \
  file:///path/script.py
```
#### **4.2 How to retain executor logs for debugging?**

The k8s would delete the pod once the executor failed in client mode and cluster mode.  If you want to get the content of executor log, you could set "temp-dir" to a mounted network file system (NFS) storage to change the log dir to replace the former one. In this case, you may meet `JSONDecodeError` because multiple executors would write logs to the same physical folder and cause conflicts. The solutions are in the next section.

```python
init_orca_context(..., extra_params = {"temp-dir": "/bigdl/"})
```

#### **4.3 How to deal with "JSONDecodeError" ?**

If you set `temp-dir` to a mounted nfs storage and use multiple executors , you may meet `JSONDecodeError` since multiple executors would write to the same physical folder and cause conflicts. Do not mount `temp-dir` to shared storage is one option to avoid conflicts. But if you debug ray on k8s, you need to output logs to a shared storage. In this case, you could set num-nodes to 1. After testing, you can remove `temp-dir` setting and run multiple executors.

#### **4.4 How to use NFS?**

If you want to save some files out of pod's lifecycle, such as logging callbacks or tensorboard callbacks, you need to set the output dir to a mounted persistent volume dir. Let NFS be a simple example.

Use NFS in client mode:

```python
init_orca_context(cluster_mode="k8s", ...,
                  conf={...,
                  "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName":"nfsvolumeclaim",
                  "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl" 
                  })
```

Use NFS in cluster mode:

```bash
${SPARK_HOME}/bin/spark-submit \
  --... ...\
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="nfsvolumeclaim" \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="/bigdl" \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName="nfsvolumeclaim" \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path="/bigdl" \
  file:///path/script.py
```

#### **4.5 How to deal with  "RayActorError" ?**

"RayActorError" may caused by running out of the ray memory. If you meet this error, try to increase the memory for ray.

```python
init_orca_context(..., exra_executor_memory_for_ray=100g)
```

####  **4.6 How to set proper "steps_per_epoch" and "validation steps" ?**

The `steps_per_epoch` and `validation_steps` should equal to numbers of dataset divided by batch size if you want to train all dataset. The `steps_per_epoch` and `validation_steps` do not relate to the `num_nodes` when total dataset and batch size are fixed. For example, you set `num_nodes` to 1, and set `steps_per_epoch` to 6. If you change the `num_nodes` to 3, the `steps_per_epoch` should still be 6.

#### **4.7 Others**

`spark.kubernetes.container.image.pullPolicy` needs to be specified as `always` if you need to update your spark executor image for k8s.

### **5. Access logs and clear pods**

When application is running, it’s possible to stream logs on the driver pod:

```bash
$ kubectl logs <spark-driver-pod>
```

To check pod status or to get some basic information around pod using:

```bash
$ kubectl describe pod <spark-driver-pod>
```

You can also check other pods using the similar way.

After finishing running the application, deleting the driver pod:

```bash
$ kubectl delete <spark-driver-pod>
```

Or clean up the entire spark application by pod label:

```bash
$ kubectl delete pod -l <pod label>
```
