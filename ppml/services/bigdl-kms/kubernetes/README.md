# Deploy BigDL KMS (Key Management Service) on Kubernetes

## BigDL KMS Architecture
![BigDLKMS](https://user-images.githubusercontent.com/60865256/211248045-dea5dac3-3169-4e02-b472-9cff901f4de7.jpg)


**bigdl-kms-frontend**: The frontend REST API provider towards KMS user.

**keywhiz**: Secret engine serving as a backend managing keys as secret.

**mysql**: Encrypted storage to save keys.

## Prerequests

- Make sure you have a workable **Kubernetes cluster/machine**
- Make sure you have a reachable **NFS**
- Prepare [bigdl-kms image](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/bigdl-kms/docker#pullbuild-container-image)

## Start BigDL KMS on Kubernetes
Modify parameters in script `install-bigdl-kms.sh`:

```
nfsServerIP: your_nfs_server_ip                   --->   <the_IP_address_of_your_NFS_server>
nfsPath: a_nfs_shared_folder_path_on_the_server   --->   <an_existing_shared_folder_path_on_NFS_server>
......
kmsIP: your_kms_ip_to_use_as                      --->   <an_unused_ip_address_in_your_subnetwork_to_assign_to_kms>
```

Then, deploy bigdl-kms on kubernetes by one command:

```bash
bash install-bigdl-kms.sh
```

Check the service whether it has successfully been running (it may take seconds).
```bash
kubectl get all -n bigdl-kms

# you will get similar to below
NAME                                      READY   STATUS    RESTARTS   AGE
pod/bigdl-kms-frontend-6d6b5f87b6-jjm76   1/1     Running   0          4m56s
pod/keywhiz-0                             2/2     Running   0          4m56s

NAME                                 TYPE           CLUSTER-IP     EXTERNAL-IP     PORT(S)          AGE
service/bigdl-kms-frontend-service   LoadBalancer   10.103.34.88   <kmsIP>         9876:31634/TCP   4m56s
service/keywhiz-service              ClusterIP      None           <none>          4444/TCP         4m56s

NAME                                 READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bigdl-kms-frontend   1/1     1            1           4m56s

NAME                                            DESIRED   CURRENT   READY   AGE
replicaset.apps/bigdl-kms-frontend-6d6b5f87b6   1         1         1       4m56s

NAME                       READY   AGE
statefulset.apps/keywhiz   1/1     4m56s
```

## Validate Status of BigDL KMS

You can communicate with BigDL KMS using client [BigDLKeyManagementService](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/kms/BigDLManagementService.scala), or simply verify through requesting REST API like below:

```
curl -k -v "https://<kmsIP>:9876/" # default port of bigdl-kms is 9876 and can be configured in bigdl-kms.yaml

# you will get similar to below
welcome to BigDL KMS Frontend

create a user like: POST /user/{userName}?token=a_token_string_for_the_user
get a primary key like: POST /primaryKey/{primaryKeyName}?user=your_username&&token=your_token
get a data key like: POST /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token
get the data key like: GET /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&&user=your_username&&token=your_token

curl -X POST -k -v "https://<kmsIP>:9876/user/<userName>?token=<userToken>"
user [<userName>] is created successfully!

curl -X POST -k -v "https://<kmsIP>:9876/primaryKey/<primaryKeyName>?user=<userName>&&token=<userToken>"
primaryKey [<primaryKeyName>] is generated successfully!

curl -X POST -k -v "https://<kmsIP>:9876/dataKey/<dataKeyName>?user=<userName>&&token=<userToken>&&primaryKeyName=<primaryKeyName>"
dataKey [<dataKeyName>] is generated successfully!

curl -X GET -k -v "https://<kmsIP>:9876/dataKey/<dataKeyName>?user=<userName>&&token=<userToken>&&primaryKeyName=<primaryKeyName>"
XY********Yw==

```

## Test KMS with PPML end-to-end example

[LocalCryptoExample](https://github.com/intel-analytics/BigDL/tree/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples#localcryptoexample-with-bigdl-kms)

[PPMLContext](https://github.com/intel-analytics/BigDL/tree/main/ppml#41-create-ppmlcontext)
