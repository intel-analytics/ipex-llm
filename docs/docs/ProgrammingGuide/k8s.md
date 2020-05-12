Analytics Zoo hyperzoo image has been built to easily run applications on Kubernetes cluster. The details of pre-installed packages and usage of the image will be introduced in this page.

- Launch pre-built hyperzoo image
- Run Analytics Zoo examples on k8s
- Run Analytics Zoo Jupyter Notebooks on remote Spark cluster or k8s
- Launch Analytics Zoo cluster serving

### **Launch pre-built hyperzoo image**

**Prerequisites**

1. Runnable docker environment has been set up.
2. A running Kubernetes cluster is prepared. Also make sure the permission of  `kubectl`  to create, list and delete pod.

**Launch pre-built hyperzoo k8s image**

1. Pull an Analytics Zoo hyperzoo image from [dockerhub](https://hub.docker.com/r/intelanalytics/hyper-zoo/tags):

```bash
sudo docker pull intelanalytics/hyper-zoo:latest
```

- Speed up pulling image by adding mirrors

To speed up pulling the image from dockerhub in China, add a registry's mirror. For Linux OS (CentOS, Ubuntu etc), if the docker version is higher than 1.12, config the docker daemon. Edit `/etc/docker/daemon.json` and add the registry-mirrors key and value:

```bash
{
  "registry-mirrors": ["https://<my-docker-mirror-host>"]
}
```

For example, add the ustc mirror in China.

```bash
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
```

Flush changes and restart docker：

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

If your docker version is between 1.8 and 1.11, find the docker configuration which location depends on the operation system. Edit and add `DOCKER_OPTS="--registry-mirror=https://<my-docker-mirror-host>"`. Restart docker `sudo service docker restart`.

If you would like to speed up pulling this image on MacOS or Windows, find the docker setting and config registry-mirrors section by specifying mirror host. Restart docker. 

Then pull the image. It will be faster.

```bash
sudo docker pull intelanalytics/hyper-zoo:latest
```

2. Launch a k8s client container:

Please note the two different containers: **client container** is for user to submit zoo jobs from here, since it contains all the required env and libs except hadoop/k8s configs; executor container is not need to create manually, which is scheduled by k8s at runtime.

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    intelanalytics/hyper-zoo:latest bash
```

Note. To launch the client container, `-v /etc/kubernetes:/etc/kubernetes:` and `-v /root/.kube:/root/.kube` are required to specify the path of kube config and installation.

To specify more argument, use:

```bash
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -e NotebookPort=12345 \
    -e NotebookToken="your-token" \
    -e http_proxy=http://your-proxy-host:your-proxy-port \
    -e https_proxy=https://your-proxy-host:your-proxy-port \
    -e RUNTIME_SPARK_MASTER=k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port> \
    -e RUNTIME_K8S_SERVICE_ACCOUNT=account \
    -e RUNTIME_K8S_SPARK_IMAGE=intelanalytics/hyper-zoo:latest \
    -e RUNTIME_PERSISTENT_VOLUME_CLAIM=myvolumeclaim \
    -e RUNTIME_DRIVER_HOST=x.x.x.x \
    -e RUNTIME_DRIVER_PORT=54321 \
    -e RUNTIME_EXECUTOR_INSTANCES=1 \
    -e RUNTIME_EXECUTOR_CORES=4 \
    -e RUNTIME_EXECUTOR_MEMORY=20g \
    -e RUNTIME_TOTAL_EXECUTOR_CORES=4 \
    -e RUNTIME_DRIVER_CORES=4 \
    -e RUNTIME_DRIVER_MEMORY=10g \
    intelanalytics/hyper-zoo:latest bash 
```

- NotebookPort value 12345 is a user specified port number.
- NotebookToken value "your-token" is a user specified string.
- http_proxy is to specify http proxy.
- https_proxy is to specify https proxy.
- RUNTIME_SPARK_MASTER is to specify spark master, which should be `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>` or `spark://<spark-master-host>:<spark-master-port>`. 
- RUNTIME_K8S_SERVICE_ACCOUNT is service account for driver pod. Please refer to k8s [RBAC](https://spark.apache.org/docs/latest/running-on-kubernetes.html#rbac).
- RUNTIME_K8S_SPARK_IMAGE is the k8s image.
- RUNTIME_PERSISTENT_VOLUME_CLAIM is to specify volume mount. We are supposed to use volume mount to store or receive data. Get ready with [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#volume-mounts).
- RUNTIME_DRIVER_HOST is to specify driver localhost (only required when submit jobs as kubernetes client mode). 
- RUNTIME_DRIVER_PORT is to specify port number (only required when submit jobs as kubernetes client mode).
- Other environment variables are for spark configuration setting. The default values in this image are listed above. Replace the values as you need.

Once the container is created, launch the container by:

```bash
sudo docker exec -it <containerID> bash
```

Then you may see it shows:

```
root@[hostname]:/opt/spark/work-dir# 
```

`/opt/spark/work-dir` is the spark work path. 

Note: The `/opt` directory contains:

- download-analytics-zoo.sh is used for downloading Analytics-Zoo distributions.
- start-notebook-spark.sh is used for starting the jupyter notebook on standard spark cluster. 
- start-notebook-k8s.sh is used for starting the jupyter notebook on k8s cluster.
- analytics-zoo-x.x-SNAPSHOT is `ANALYTICS_ZOO_HOME`, which is the home of Analytics Zoo distribution.
- analytics-zoo-examples directory contains downloaded python example code.
- jdk is the jdk home.
- spark is the spark home.
- redis is the redis home.

### **Run Analytics Zoo examples on k8s**

**Launch an Analytics Zoo python example on k8s**

Here is a sample for submitting the python [anomalydetection](https://github.com/intel-analytics/analytics-zoo/tree/master/pyzoo/zoo/examples/anomalydetection) example on cluster mode.

```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip,/opt/analytics-zoo-examples/python/anomalydetection/anomaly_detection.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  file:///opt/analytics-zoo-examples/python/anomalydetection/anomaly_detection.py \
  --input_dir /zoo/data/nyc_taxi.csv
```

Options:

- --master: the spark mater, must be a URL with the format `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`. 
- --deploy-mode: submit application in cluster mode or client mode.
- --name: the Spark application name.
- --conf: require to specify k8s service account, container image to use for the Spark application, driver volumes name and path, label of pods, spark driver and executor configuration, etc.
  check the argument settings in your environment and refer to the [spark configuration page](https://spark.apache.org/docs/latest/configuration.html) and [spark on k8s configuration page](https://spark.apache.org/docs/latest/running-on-kubernetes.html#configuration) for more details.
- --properties-file: the customized conf properties.
- --py-files: the extra python packages is needed.
- file://: local file path of the python example file in the client container.
- --input_dir: input data path of the anomaly detection example. The data path is the mounted filesystem of the host. Refer to more details by [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes).

**Launch an Analytics Zoo scala example on k8s**

Here is a sample for submitting the scala [anomalydetection](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/anomalydetection) example on cluster mode

```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar \
  --class com.intel.analytics.zoo.examples.anomalydetection.AnomalyDetection \
  ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip \
  --inputDir /zoo/data
```

Options:

- --master: the spark mater, must be a URL with the format `k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`. 
- --deploy-mode: submit application in cluster mode or client mode.
- --name: the Spark application name.
- --conf: require to specify k8s service account, container image to use for the Spark application, driver volumes name and path, label of pods, spark driver and executor configuration, etc.
  check the argument settings in your environment and refer to the [spark configuration page](https://spark.apache.org/docs/latest/configuration.html) and [spark on k8s configuration page](https://spark.apache.org/docs/latest/running-on-kubernetes.html#configuration) for more details.
- --properties-file: the customized conf properties.
- --py-files: the extra python packages is needed.
- --class: scala example class name.
- --input_dir: input data path of the anomaly detection example. The data path is the mounted filesystem of the host. Refer to more details by [Kubernetes Volumes](https://spark.apache.org/docs/latest/running-on-kubernetes.html#using-kubernetes-volumes).

**Access logs to check result and clear pods**

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

### **Run Analytics Zoo Jupyter Notebooks on remote Spark cluster or k8s**

When started a Docker container with specified argument RUNTIME_SPARK_MASTER=`k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>` or RUNTIME_SPARK_MASTER=`spark://<spark-master-host>:<spark-master-port>`, the container will submit jobs to k8s cluster or spark cluster if you use $RUNTIME_SPARK_MASTER as url of spark master.

You may also need to specify NotebookPort=`<your-port>` and NotebookToken=`<your-token>` to start Jupyter Notebook on the specified port and bind to 0.0.0.0.

To start the Jupyter notebooks on remote spark cluster, please use RUNTIME_SPARK_MASTER=`spark://<spark-master-host>:<spark-master-port>`, and attach the client container with command: “docker exec -it `<container-id>`  bash”, then run the shell script: “/opt/start-notebook-spark.sh”, this will start a Jupyter notebook instance on local container, and each tutorial in it will be submitted to the specified spark cluster. User can access the notebook with url `http://<local-ip>:<your-port>` in a preferred browser, and also need to input required  token with `<your-token>` to browse and run the tutorials of Analytics Zoo. Each tutorial will run driver part code in local container and run executor part code on spark cluster.

To start the Jupyter notebooks on Kubernetes cluster, please use RUNTIME_SPARK_MASTER=`k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`, and attach the client container with command: “docker exec -it `<container-id>`  bash”, then run the shell script: “/opt/start-notebook-k8s.sh”, this will start a Jupyter notebook instance on local container, and each tutorial in it will be submitted to the specified kubernetes cluster. User can access the notebook with url `http://<local-ip>:<your-port>` in a preferred browser, and also need to input required  token with `<your-token>` to browse and run the tutorials of Analytics Zoo. Each tutorial will run driver part code in local container and run executor part code in dynamic allocated spark executor pods on k8s cluster. 

### **Launch Analytics Zoo cluster serving**

To run Analytics Zoo cluster serving in hyper-zoo client container and submit the streaming job on K8S cluster, you may need to specify arguments RUNTIME_SPARK_MASTER=`k8s://https://<k8s-apiserver-host>:<k8s-apiserver-port>`, and you may also need to mount volume from host to container to load model and data files.

You can leverage an existing Redis instance/cluster, or you can start one in the client container:
```bash
${REDIS_HOME}/src/redis-server ${REDIS_HOME}/redis.conf > ${REDIS_HOME}/redis.log &
```
And you can check the running logs of redis:
```bash
cat ${REDIS_HOME}/redis.log
```

Before starting the cluster serving job, please also modify the config.yaml to configure correct path of the model and redis host url, etc.
```bash
nano /opt/cluster-serving/config.yaml
```

After that, you can start the cluster-serving job and submit the streaming job on K8S cluster:
```bash
${SPARK_HOME}/bin/spark-submit \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=${RUNTIME_K8S_SERVICE_ACCOUNT} \
  --name analytics-zoo \
  --conf spark.kubernetes.container.image=${RUNTIME_K8S_SPARK_IMAGE} \
  --conf spark.executor.instances=${RUNTIME_EXECUTOR_INSTANCES} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.driver.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.options.claimName=${RUNTIME_PERSISTENT_VOLUME_CLAIM} \
  --conf spark.kubernetes.executor.volumes.persistentVolumeClaim.${RUNTIME_PERSISTENT_VOLUME_CLAIM}.mount.path=/zoo \
  --conf spark.kubernetes.driver.label.<your-label>=true \
  --conf spark.kubernetes.executor.label.<your-label>=true \
  --executor-cores ${RUNTIME_EXECUTOR_CORES} \
  --executor-memory ${RUNTIME_EXECUTOR_MEMORY} \
  --total-executor-cores ${RUNTIME_TOTAL_EXECUTOR_CORES} \
  --driver-cores ${RUNTIME_DRIVER_CORES} \
  --driver-memory ${RUNTIME_DRIVER_MEMORY} \
  --properties-file ${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf \
  --py-files ${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-python-api.zip,/opt/analytics-zoo-examples/python/anomalydetection/anomaly_detection.py \
  --conf spark.driver.extraJavaOptions=-Dderby.stream.error.file=/tmp \
  --conf spark.sql.catalogImplementation='in-memory' \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar:/opt/cluster-serving/spark-redis-2.4.0-jar-with-dependencies.jar \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_HOME}/lib/analytics-zoo-bigdl_${BIGDL_VERSION}-spark_${SPARK_VERSION}-${ANALYTICS_ZOO_VERSION}-jar-with-dependencies.jar:/opt/cluster-serving/spark-redis-2.4.0-jar-with-dependencies.jar \
  --conf "spark.executor.extraJavaOptions=-Dbigdl.engineType=mklblas" \
  --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mklblas" \
  --class com.intel.analytics.zoo.serving.ClusterServing \
  local:/opt/analytics-zoo-0.8.0-SNAPSHOT/lib/analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.8.0-SNAPSHOT-jar-with-dependencies.jar
```
