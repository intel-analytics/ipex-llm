# Use zeppelin in bigdl-ppml
Apache Zeppelin is web-based notebook that enables data-driven, interactive data analytics and collaborative documents with SQL, Scala, Python, R and more.
We can use zeppelin in bigdl-ppml through the following steps.
## Prepare Docker Images
We will use two docker images, so we need to prepare them first:
```bash
docker pull intelanalytics/zeppelin-interpreter:0.11.0-SNAPSHO
docker pull intelanalytics/zeppelin-server:0.11.0-SNAPSHOT
```
## Deploy zeppelin on k8s in bigdl-ppml container
Since there is no kubectl in the Docker container, we use the curl command to operate the k8s API to deploy Zeppelin.
### Prepare ServiceAccount 
We need to prepare a service account with sufficient permissions for authentication in the subsequent curl commands.
We recommend using the provided `zeppelin-service-account.yaml` to create the service account.
You should first move the `zeppelin-service-account.yaml` file to the master node of your Kubernetes cluster.
```bash
# In k8s master node
kubectl apply -f zeppelin-service-account.yaml
```
### Prepare relevant environment variables
Before execute `deploy.sh`, we need to prepare relevant environment variables.
First, in your k8s master node:
```bash
export CLUSTER_NAME="kubernetes"
APISERVER=$(kubectl config view -o jsonpath="{.clusters[?(@.name==\"$CLUSTER_NAME\")].cluster.server}")
echo $APISERVER 
TOKEN=$(kubectl get secrets -o jsonpath="{.items[?(@.metadata.annotations['kubernetes\.io/service-account\.name']=='zeppelin')].data.token}"|base64 -d)
echo $TOKEN
```
Then, in your bigdl-ppml container:
```bash
docker exec -it <bigdl-ppml container name> bash #enter your bigdl-ppml container
export APISERVER=<output of echo $APISERVER in the previous step>
export TOKEN=<output of echo $TOKEN in the previous step>
```
### Deploy and delete zeppelin

```bash
# Make sure you are inside the container.
cd /ppml/zeppelin
chmod +x deploy.sh
./deploy.sh  # zeppelin deployment is now completed

./delete.sh  # Delete zeppelin deployment
```
