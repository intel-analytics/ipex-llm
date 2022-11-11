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
If you already have a BidDL-PCCS service on Kubernetes, please skip this step.

If not, please **[deploy BigDL-PCCS on Kubernetes](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/pccs/kubernetes)**.
## 2. Start BigDL-eHSM-KMS on Kubernetes 

Please make sure current workdir is `kubernetes`.

Then modify parameters in `install-bigdl-ehsm-kms.sh` as following:

```shell
# reset of other parameters in values.yaml is optional, please check according to your environment
nfsServerIP: your_nfs_server_ip                   --->   <the_IP_address_of_your_NFS_server>
nfsPath: a_nfs_shared_folder_path_on_the_server   --->   <an_existing_shared_folder_path_on_NFS_server>
......
pccsIP: your_pccs_ip                              --->   <an_used_ip_address_in_your_subnetwork_you_have_assigned_to_pccs_in_step1>
dkeyserverIP: your_dkeyserver_ip_to_use_as        --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_dkeyserver>
kmsIP: your_kms_ip_to_use_as                      --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_kms>

# Replace the below parameters according to your environment
apiKey: your_intel_pcs_server_subscription_key_obtained_through_web_registeration
countryName: your_country_name
cityName: your_city_name
organizaitonName: your_organizaition_name
commonName: server_fqdn_or_your_name
```

Then, deploy BigDL-eHSM-KMS on kubernetes:

```bash
bash install-bigdl-ehsm-kms.sh
```

Check the service whether it has successfully been running (it may take seconds):

```bash
kubectl get all -n bigdl-ehsm-kms

# you will get similar to below
NAME                                                   READY   STATUS    RESTARTS   AGE
pod/bigdl-ehsm-kms-deployment-7dd7c965d5-bqj9t         1/1     Running   0          6h52m
pod/couchdb-0                                          1/1     Running   0          6h52m
pod/dkeycache-57db49f98-z28t4                          1/1     Running   0          6h52m
pod/dkeyserver-0                                       1/1     Running   0          6h52m

NAME                                   TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)          AGE
service/bigdl-ehsm-kms-service         LoadBalancer   1.10.9.98       1.1.0.218       9000:30000/TCP   6h52m
service/couchdb                        ClusterIP      1.10.8.236      <none>          5984/TCP         6h52m
service/dkeyserver                     ClusterIP      1.10.1.132      1.1.0.217       8888/TCP         6h52m

NAME                                              READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bigdl-ehsm-kms-deployment         1/1     1            1           6h52m
deployment.apps/dkeycache                         1/1     1            1           6h52m

NAME                                                         DESIRED   CURRENT   READY   AGE
replicaset.apps/bigdl-ehsm-kms-deployment-7dd7c965d5   1         1         1       6h52m
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
namespace "bigdl-ehsm-kms" deleted
```
