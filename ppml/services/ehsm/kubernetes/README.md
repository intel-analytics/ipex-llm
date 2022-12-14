# Deploy BigDL-eHSM-KMS on Kubernetes



![ehsm](https://user-images.githubusercontent.com/108786898/197957072-a1574387-3dbb-4cda-be52-4bb5fcb06d0c.png)


**pccs** provides SGX attestation support service.

**dkeyserver** generates domain keys inside enclave.

**dkeycache** caches keys and thus increases key hit ratio. It communicates with dkeyserver based on pccs.

**couchdb** stores run-time data (like enroll info), the persistent is based on Kubernetes PV and NFS. 

**bigdl-ehsm-kms-deployment** is the direct key service provider.


## Prerequests

- Please make sure you have a workable **Kubernetes cluster/machine**.
- Please make sure you have a usable https proxy.
- Please make sure your **CPU** is able to run PCCS service, which generate and verify quotes.
- Please make sure you have a reachable **NFS**.
- Please make sure you have an usable PCCS ApiKey for your platform. The PCCS uses this API key to request collaterals from Intel's Provisioning Certificate Service. User needs to subscribe first to obtain an API key. For how to subscribe to Intel Provisioning Certificate Service and receive an API key, goto https://api.portal.trustedservices.intel.com/provisioning-certification and click on 'Subscribe'.



Now you have already had a PCCS image.


## 1. Deploy BigDL-PCCS on Kubernetes
If you already have a BigDL-PCCS service on Kubernetes, please skip this step.

If not, please **[deploy BigDL-PCCS on Kubernetes](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/pccs/kubernetes)**.
## 2. Start BigDL-eHSM-KMS on Kubernetes 
### 2.1 Determine IP addresses for dkeyserver and KMS
First of all , **IP address that unused in your subnetwork** is needed as KMS external service IP, and it should be different from PCCS IP that you have set in step 1. \
**Especially,** the IP addresses chosen for dkeyserver and KMS **SHOULD NOT** be real machine IP address. \
You could check if the IP adresses are available for dkeyserver and KMS like this
```bash
# assume your IP address is 1.2.3.4, and you want to use 1.2.3.227 as dkeyserver IP
ping 1.2.3.227

# information below means 1.2.3.227 is expected to be an appropriate IP addess for dkeyserver. 
# otherwise, you are supposed to test another one.
PING 1.2.3.227 (1.2.3.227) 56(84) bytes of data.
From 1.2.3.4 icmp_seq=1 Destination Host **Unreachable**
........

# try another IP address (e.g 1.2.3.228) for KMS with the same approach.
```

### 2.2 Modify the script and deploy BigDL-eHSM-KMS
Please make sure current workdir is `kubernetes`.

Then modify parameters in `install-bigdl-ehsm-kms.sh` as following. \
The `pccsIP` should be the IP address you have used in step 1. The `dkeyserverIP` and `kmsIP` should be the IP addresses you have determined in step 2.1. 

```shell
# reset of other parameters in values.yaml is optional, please check according to your environment
nfsServerIP: your_nfs_server_ip                   --->   <the_IP_address_of_your_NFS_server>
nfsPath: a_nfs_shared_folder_path_on_the_server   --->   <an_existing_shared_folder_path_on_NFS_server>
......
pccsIP: your_pccs_ip                              --->   <the_ip_address_in_your_subnetwork_you_have_assigned_to_pccs_in_step1>
kmsIP: your_kms_ip_to_use_as                      --->   <an_unused_ip_address_in_your_subnetwork_to_assign_to_kms>
dkeyserverNodeName: the_fixed_node_you_want_to_assign_dkeyserver_to --->   <a_node_name_in_k8s_cluster_to_run_dkerysever_on>
```

Then, deploy BigDL-eHSM-KMS on kubernetes:

```bash
bash install-bigdl-ehsm-kms.sh
```

Check the service whether it has successfully been running (it may take seconds). \
**Note:** the `EXTERNAL-IP` of `service/bigdl-ehsm-kms-service` is the **KMS IP**. 

```bash
kubectl get all -n bigdl-ehsm-kms

# you will get similar to below
NAME                                                   READY   STATUS    RESTARTS   AGE
pod/bigdl-ehsm-kms-deployment-7dd7c965d5-bqj9t         1/1     Running   0          6h52m
pod/couchdb-0                                          1/1     Running   0          6h52m
pod/dkeycache-57db49f98-z28t4                          1/1     Running   0          6h52m
pod/dkeyserver-0                                       1/1     Running   0          6h52m

NAME                             TYPE           CLUSTER-IP       EXTERNAL-IP     PORT(S)          AGE
service/bigdl-ehsm-kms-service   LoadBalancer   10.103.152.224   172.168.0.238   9000:30011/TCP   56m
service/couchdb                  ClusterIP      10.103.152.212   <none>          5984/TCP         56m
service/dkeyserver               ClusterIP      10.103.152.227   <none>          8888/TCP         56m

NAME                                              READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bigdl-ehsm-kms-deployment         1/1     1            1           6h52m
deployment.apps/dkeycache                         1/1     1            1           6h52m

NAME                                                         DESIRED   CURRENT   READY   AGE
replicaset.apps/bigdl-ehsm-kms-deployment-7dd7c965d5         1         1         1       6h52m
replicaset.apps/dkeycache-57db49f98                          1         1         1       6h52m

NAME                          READY   AGE
statefulset.apps/couchdb      1/1     6h52m
statefulset.apps/dkeyserver   1/1     6h52m


# Check the status of KMS
curl -v -k -G "https://<external_kms_ip>:9000/ehsm/?Action=GetVersion"

{"code":200,"message":"success!","result":{"git_sha":"5c91d6d","version":"0.2.1"}}

```

## 3. Enroll through RestAPI

```bash
curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"

......

{"code":200,"message":"successful","result":{"apikey":"E8QKpBBapaknprx44FaaTY20rptg54Sg","appid":"8d5dd3b8-3996-40f5-9785-dcb8265981ba"}}
```

## 4. Test BigDL-eHSM-KMS with SimpleQuerySparkExample

Test with following scala spark example:
### [SimpleQuerySparkExample](https://github.com/intel-analytics/BigDL/tree/main/ppml#step-0-preparation-your-environment) 



## 5. Delete Service from Kubernetes

You can quickly and easily delete BigDL-eHSM-KMS from Kubernetes with following commands:

```bash
bash uninstall-bigdl-ehsm-kms.sh

# you will get similar to below if success
service "couchdb" deleted
service "bigdl-ehsm-kms-service" deleted
service "dkeyserver" deleted
deployment.apps "bigdl-ehsm-kms-deployment" deleted
deployment.apps "dkeycache" deleted
statefulset.apps "couchdb" deleted
statefulset.apps "dkeyserver" deleted
persistentvolumeclaim "couch-persistent-storage-couchdb-0" deleted
persistentvolume "ehsm-pv-nfs" deleted
namespace "bigdl-ehsm-kms" deleted
```
