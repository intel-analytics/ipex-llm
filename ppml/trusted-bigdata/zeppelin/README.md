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

### Deploy zeppelin

```bash
# Get token on the host machine
TOKEN=$(kubectl get secrets -o jsonpath="{.items[?(@.metadata.annotations['kubernetes\.io/service-account\.name']=='zeppelin')].data.token}"|base64 -d)
echo $TOKEN
# Make sure you are inside the container.
docker exec -it <bigdl-ppml container name> bash #enter your bigdl-ppml container
cd /ppml/zeppelin
# Set relevant environment variables.
export TOKEN=<output of echo $TOKEN in the previous step>
./deploy.sh  # zeppelin deployment is now completed
```
## Access Zeppelin web ui
Port forward Zeppelin server port,
```bash
kubectl port-forward $(kubectl get pods -o name | grep zeppelin-server | head -n 1 | sed 's\pod/\\') 8080:80 --address 0.0.0.0
```
and browse localhost:8080. 

## Delete zeppelin 
```bash
# Get token on the host machine
TOKEN=$(kubectl get secrets -o jsonpath="{.items[?(@.metadata.annotations['kubernetes\.io/service-account\.name']=='zeppelin')].data.token}"|base64 -d)
echo $TOKEN
# Make sure you are inside the container.
docker exec -it <bigdl-ppml container name> bash #enter your bigdl-ppml container
cd /ppml/zeppelin
# Set relevant environment variables.
export TOKEN=<output of echo $TOKEN in the previous step>
./delete.sh  # zeppelin deployment is now deleted
```
