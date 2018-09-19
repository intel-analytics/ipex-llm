Spark can run on clusters managed by Kubernetes. This feature makes use of native Kubernetes scheduler that has been added to Spark 2.3. Now as a deep learning library for Apache
Spark, BigDL can also run on Kubernetes by leveraging Spark on Kubernetes.

---

## **Prerequisites**

- You need to have a running Kubernetes cluster that support Spark on Kubernetes. See [here](https://spark.apache.org/docs/2.3.0/running-on-kubernetes.html#prerequisites)
Otherwise, you can use [minikube](https://kubernetes.io/docs/setup/minikube/) to run kubernetes locally and start a test. 

- Download the [spark2.3](https://spark.apache.org/downloads.html) release from Spark and unzip it. 

---

## **Docker image**
For Spark2.3,
BigDL ships with a Dockerfile that can be found in the ```docker/spark2.3-k8s``` directory. 

To built it, copy the 
``docker/spark2.3-k8s`` folder under ```kubernetes/dockerfiles/``` your unzipped spark 2.3 folder. 

Then the  docker build 
command should be invoked from the top level directory of the Spark distribution. E.g.:
```docker build -t bigdl-spark2.3-k8s:latest -f kubernetes/dockerfiles/spark2.3-k8s/Dockerfile . ```

You can set your own image name and tag. We'll just use ```bigdl-spark2.3-k8s:latest``` as the name and tag for
demonstration in the following usage example.

---

## **Run BigDL examples**

Now, let's go on a quick tour on how to run BigDL Lenet5 example with a local k8s cluster created by minikube.

### Build your BigDL on Kubernetes image
Follow the instructions under **Docker Image** section above, and you can have your pre-built docker image 
```bigdl-spark2.3-k8s:latest```.

Now you can tag it and push it to your docker hub for your k8s cluster's docker 
 deamon to pull it. Something like:
```$shell
# tag the image for yourself
docker tag bigdl-spark2.3-k8s my-repo-name/bigdl-spark2.3-k8s 
# push
docker push my-repo-name/bigdl-spark2.3-k8s
```
Or If we test with minikube, you can build your docker image using minikube since it will do so 
directly into minikube's Docker daemon. There is no need to push the images into minikube in that 
case, they'll be automatically available when running applications inside the minikube cluster.

### Download minikube
Follow the instructions from the [minikube chapter](https://kubernetes.io/docs/tasks/tools/install-minikube/) 
in the Kubernetes official document and install minikube on your client machine.

After installation, run ```minikube start```. You can see the terminal output like this:
```shell
Starting local Kubernetes v1.10.0 cluster...
Starting VM...
Getting VM IP address...
Moving files into cluster...
Setting up certs...
Connecting to cluster...
Setting up kubeconfig...
Starting cluster components...
Kubectl is now configured to use the cluster.
Loading cached images from config file.

```
To further verify your minikube's local k8s cluster is correctly launched, run ```kubetcl cluster-info```.
It tells you where the kubernetes master DNS is running at. Sample output should be like:
```shell
Kubernetes master is running at https://192.168.99.100:8443
KubeDNS is running at https://192.168.99.100:8443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```

### create Kubernetes service account
```shell
kubectl create spark(you can replace it with your preferred name)
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```
The reason why we need to do this is illustrated [here](https://spark.apache.org/docs/2.3.0/running-on-kubernetes.html#rbac).
In simple words, the ```default``` service account may not allow driver pods to create pods so
we need to create another one granted with access.

### Run the Letnet5 application
It's been a lot of setup work that might make your palm sweaty but finally we are ready to launch the application.

Run the script below to train lenet5 model for only 2 epochs on MNIST dataset as a demo. 
```shell
SPARK_HOME=...(your spark 2.3 home directory)
$SPARK_HOME/bin/spark-submit \
    --master k8s://your-master-ip(found by kubertcl cluster-info)\
    --deploy-mode cluster \
    --name bigdl-lenet5 \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark(Or your own service account name) \
    --conf spark.executor.instances=4 \
    --conf spark.executor.cores=1 \
    --conf spark.cores.max=4 \
    --conf spark.kubernetes.container.image=docker.io/my-repo-name/bigdl-spark2.3-k8s:latest \
    local:///opt/bigdl-0.6.0/lib/bigdl-SPARK_2.3-0.6.0-jar-with-dependencies.jar \
    -f hdfs://path-to-your-mnist \
    -b 128 \
    -e 2 \
    --checkpoint /tmp
```

In the above commands
* -f: where you put your MNIST data
* --checkpoint: Where you cache the model/train_state snapshot. You should input a folder and
make sure the folder is created when you run this example. The model snapshot will be named as
model.#iteration_number, and train state will be named as state.#iteration_number. Note that if
there are some files already exist in the folder, the old file will not be overwrite for the
safety of your model files.
* -b: The mini-batch size. It is expected that the mini-batch size is a multiple of node_number * core_number.

You can find more information about this example in 
BigDL [lenet](https://github.com/intel-analytics/BigDL/tree/master/pyspark/bigdl/models/lenet) folder.

---

## **Future Work**
As there are several Spark on Kubernetes features that await to be added into the future versions of Spark. See more 
[here](https://spark.apache.org/docs/2.3.0/running-on-kubernetes.html#future-work). Some of those such as 
Pyspark and Local File Dependency Management BigDL will also support by leveraging Spark, and the document, docker 
image plus script to run the examples so please stay tuned. We welcome you send your questions in our BigDL user group 
during the usage.
