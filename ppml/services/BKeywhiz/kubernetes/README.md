# Deploy BKeywhiz (BigDL Keywhiz KMS) on Kubernetes

## BigDL KMS Architecture
![BigDLKMS](https://user-images.githubusercontent.com/60865256/207252206-d7eeff16-5174-470a-bbda-262db8f39ca1.jpg)

**bkeywhiz-kms-frontend**: The frontend REST API provider towards KMS user.

**keywhiz**: Crypto engine serving as a backend key crypto codec.

**mysql**: Encrypted storage to save keys.

## Prerequests

- Please make sure you have a workable **Kubernetes cluster/machine**.
- Please make sure you have a reachable **NFS**.

## Start BKeywhiz on Kubernetes
Modify parameters in script `install-bkeywhiz-kms.sh`:

```
nfsServerIP: your_nfs_server_ip                   --->   <the_IP_address_of_your_NFS_server>
nfsPath: a_nfs_shared_folder_path_on_the_server   --->   <an_existing_shared_folder_path_on_NFS_server>
......
kmsIP: your_kms_ip_to_use_as                      --->   <an_unused_ip_address_in_your_subnetwork_to_assign_to_kms>
```

Then, deploy BKeywhiz on kubernetes by one command:

```bash
bash install-bkeywhiz-kms.sh
```

Check the service whether it has successfully been running (it may take seconds).
```bash
kubectl get all -n bkeywhiz

# you will get similar to below
NAME                                     READY   STATUS    RESTARTS   AGE
pod/bkeywhiz-frontend-6d6b5f87b6-jjm76   1/1     Running   0          4m56s
pod/keywhiz-0                            2/2     Running   0          4m56s

NAME                                TYPE           CLUSTER-IP     EXTERNAL-IP     PORT(S)          AGE
service/bkeywhiz-frontend-service   LoadBalancer   10.103.34.88   <kmsIP>         9876:31634/TCP   4m56s
service/keywhiz-service             ClusterIP      None           <none>          4444/TCP         4m56s

NAME                                READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bkeywhiz-frontend   1/1     1            1           4m56s

NAME                                           DESIRED   CURRENT   READY   AGE
replicaset.apps/bkeywhiz-frontend-6d6b5f87b6   1         1         1       4m56s

NAME                       READY   AGE
statefulset.apps/keywhiz   1/1     4m56s
```

You can communicate with BKeywhiz KMS using client [BKeywhizKeyManagementService](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/kms/BKeywhizManagementService.scala), or simply verify through requesting REST API like below:

```
curl -k -v "https://<kmsIP>:9876/" # default port of BKeywhiz is 9876 and can be configured in bkeywhiz-kms.yaml

# you will get similar to below
welcome to BigDL Keywhiz KMS Frontend

create a user like: POST /user/{userName}?password=a_password_for_the_user
get a primary key like: POST /primaryKey/{primaryKeyName}?user=your_username&password=your_password
get a data key like: POST /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&user=your_username&password=your_password
get the data key like: GET /dataKey/{dataKeyName}?primaryKeyName=the_primary_key_name&user=your_username&password=your_password

curl -X POST -k -v "https://<kmsIP>:9876/user/<userName>?password=<userPassword>"
user [<userName>] is created successfully!

curl -X POST -k -v "https://<kmsIP>:9876/primaryKey/<primaryKeyName>?user=<userName>&&password=<userPassword>"
primaryKey [<primaryKeyName>] is generated successfully!

......

```
