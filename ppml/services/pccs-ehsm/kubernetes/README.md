# Deploy BigDL-PCCS-eHSM-KMS on Kubernetes with Helm Charts


![bigdl-pccs-ehsm-kms-architecture2 0](https://user-images.githubusercontent.com/60865256/174554804-4c0b361b-413b-48f9-bddc-dc2d1ad5b81e.png)


**pccs** provides SGX attestation support service.

**dkeyserver** generates domain keys inside enclave.

**dkeycache** caches keys and thus increases key hit ratio. It communicates with dkeyserver based on pccs.

**couchdb** stores run-time data (like enroll info), the persistent is based on Kubernetes PV and NFS. 

**bigdl-pccs-ehsm-kms-deployment** is the direct key service provider.


## Prerequests

- Please make sure you have a workable **Kubernetes cluster/machine**.
- Please make sure you have a usable https proxy.
- Please make sure your **CPU** is able to run PCCS service, which generate and verify quotes.
- Please make sure you have a reachable **NFS**.
- Please make sure you have already installed **[helm](https://helm.sh/)**.
- Please make sure you have an usable PCCS ApiKey for your platform. The PCCS uses this API key to request collaterals from Intel's Provisioning Certificate Service. User needs to subscribe first to obtain an API key. For how to subscribe to Intel Provisioning Certificate Service and receive an API key, goto https://api.portal.trustedservices.intel.com/provisioning-certification and click on 'Subscribe'.

## 1. Pull/Build the PCCS Image

We encapsulate host PCCS service into a docker image, which enables a user-friendly container-service.

Download image as below:

```bash
docker pull intelanalytics/pccs:0.3.0-SNAPSHOT
```

Or you are allowed to build the image manually:

```bash
cd ../pccs
# configure build parameters in build-docker-image.sh
bash build-docker-image.sh
cd ../kubernetes
```

Now you have already had a PCCS image.


## 2. Start BigDL-PCCS-eHSM-KMS on Kubernetes 

Please make sure current workdir is `kubernetes`.

Then modify parameters in `values.yaml` as following:

```shell
# reset of other parameters in values.yaml is optional, please check according to your environment
nfsServerIP: your_nfs_server_ip                   --->   <the_IP_address_of_your_NFS_server>
nfsPath: a_nfs_shared_folder_path_on_the_server   --->   <an_existing_shared_folder_path_on_NFS_server>
......
pccsIP: your_pccs_ip_to_use_as                    --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_pccs>
dkeyserverIP: your_dkeyserver_ip_to_use_as        --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_dkeyserver>
kmsIP: your_kms_ip_to_use_as                      --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_kms>

# Replace the below parameters according to your environment

apiKey: your_intel_pcs_server_subscription_key_obtained_through_web_registeration
httpsProxyUrl: your_usable_https_proxy_url
countryName: your_country_name
cityName: your_city_name
organizaitonName: your_organizaition_name
commonName: server_fqdn_or_your_name
```

Then, deploy BigDL-PCCS-eHSM-KMS on kubernetes:

```bash
kubectl create namespace bigdl-pccs-ehsm-kms
helm install kms . # kms can be modified to any name as you like
```

Check the service whether it has successfully been running (it may take seconds):

```bash
kubectl get all -n bigdl-pccs-ehsm-kms

# you will get similar to below
NAME                                                  READY   STATUS    RESTARTS   AGE
pod/bigdl-pccs-ehsm-kms-deployment-8548fd854c-wvr8f   1/1     Running   0          4h57m
pod/couchdb-0                                         1/1     Running   0          4h57m
pod/dkeycache-86f8b5456b-8ml57                        1/1     Running   0          4h57m
pod/dkeyserver-0                                      1/1     Running   0          4h57m
pod/pccs-0                                            1/1     Running   0          4h57m

NAME                                  TYPE           CLUSTER-IP       EXTERNAL-IP           PORT(S)          AGE
service/bigdl-pccs-ehsm-kms-service   LoadBalancer   10.103.155.211   <kms_external_ip>     9000:30000/TCP   4h57m
service/couchdb                       ClusterIP      10.97.91.172     <none>                5984/TCP         4h57m
service/dkeyserver                    ClusterIP      10.98.66.45      1.2.4.114             8888/TCP         4h57m
service/pccs                          ClusterIP      10.98.213.70     1.2.4.115             18081/TCP        4h57m

NAME                                             READY   UP-TO-DATE   AVAILABLE   AGE
deployment.apps/bigdl-pccs-ehsm-kms-deployment   1/1     1            1           4h57m
deployment.apps/dkeycache                        1/1     1            1           4h57m

NAME                                                        DESIRED   CURRENT   READY   AGE
replicaset.apps/bigdl-pccs-ehsm-kms-deployment-8548fd854c   1         1         1       4h57m
replicaset.apps/dkeycache-86f8b5456b                        1         1         1       4h57m

NAME                          READY   AGE
statefulset.apps/couchdb      1/1     4h57m
statefulset.apps/dkeyserver   1/1     4h57m
statefulset.apps/pccs         1/1     4h57m

# Check the status of KMS
curl -v -k -G "https://<external_kms_ip>:9000/ehsm/?Action=GetVersion"

{"code":200,"message":"success!","result":{"git_sha":"5c91d6d","version":"0.2.1"}}

```

## 3. Enroll

### Scheme 1: through kms-utils container
We encapsulate eHSM enroll into a docker image, you can pull it or build by yourself like below:

```bash
docker pull intelanalytics/kms-utils:0.3.0-SNAPSHOT
# OR

cd ../kms-utils
# Please edit parameters inside build-docker-image.sh first
bash build-docker-image.sh
```

If image is ready, you can run the container and enroll by using `run-docker-container.sh` in order to get a appid and apikey pair like below:

```bash
export KMS_TYPE=an_optional_kms_type # KMS_TYPE can be (1) ehsm, (2) simple
export EHSM_KMS_IP=your_ehsm_kms_ip # if ehsm
export EHSM_KMS_PORT=your_ehsm_kms_port # if ehsm
export ENROLL_IMAGE_NAME=your_enroll_image_name_built
export ENROLL_CONTAINER_NAME=your_enroll_container_name_to_run
export PCCS_URL=your_pccs_url # format like https://x.x.x.x:xxxx/sgx/certification/v3/

sudo docker run -itd \
    --privileged \
    --net=host \
    --name=$ENROLL_CONTAINER_NAME \
    -v /dev/sgx/enclave:/dev/sgx/enclave \
    -v /dev/sgx/provision:/dev/sgx/provision \
    -v $local_data_folder_path:/home/data \
    -v $local_key_folder_path:/home/key \
    -e EHSM_KMS_IP=$EHSM_KMS_IP \
    -e EHSM_KMS_PORT=$EHSM_KMS_PORT \
    -e KMS_TYPE=$KMS_TYPE \
    -e PCCS_URL=$PCCS_URL
    $ENROLL_IMAGE_NAME bash
    
docker exec -i $ENROLL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh enroll"
INFO [main.cpp(46) -> main]: ehsm-kms enroll app start.
INFO [main.cpp(86) -> main]: First handle:  send msg0 and get msg1.
INFO [main.cpp(99) -> main]: First handle success.
INFO [main.cpp(101) -> main]: Second handle:  send msg2 and get msg3.
INFO [main.cpp(118) -> main]: Second handle success.
INFO [main.cpp(120) -> main]: Third handle:  send att_result_msg and get ciphertext of the APP ID and API Key.

appid: d792478c-f590-4073-8ed6-2d15e714da78

apikey: bSMN3dAQGEwgx297Ff1H2umBzwzv6W34

INFO [main.cpp(155) -> main]: decrypt APP ID and API Key success.
INFO [main.cpp(156) -> main]: Third handle success.
INFO [main.cpp(159) -> main]: ehsm-kms enroll app end.
```

### Scheme 2: through ehsm-kms_enroll_app

Since only the user with valid APPID and APIKey could request the public cryptographic restful APIs, eHSM-KMS provides a new Enroll APP which is used to retrieve the APPID and APIKey from the eHSM-core enclave via the remote secure channel (based on the SGX remote attestation).

First, clone the eHSM project:

```bash
git clone https://github.com/intel/ehsm.git
```

Compile and get the executable ehsm-kms_enroll_app file:

```bash
sudo apt update

sudo apt install vim autoconf automake build-essential cmake curl debhelper git libcurl4-openssl-dev libprotobuf-dev libssl-dev libtool lsb-release ocaml ocamlbuild protobuf-compiler wget libcurl4 libssl1.1 make g++ fakeroot libelf-dev libncurses-dev flex bison libfdt-dev libncursesw5-dev pkg-config libgtk-3-dev libspice-server-dev libssh-dev python3 python3-pip  reprepro unzip libjsoncpp-dev uuid-dev

cd ehsm
make
cd out/ehsm-kms_enroll_app
ls ehsm-kms_enroll_app
```

Then, you will find a new target file `ehsm-kms_enroll_app` generated.

Now, you can enroll your app through command below, and you will receive a appid-apikey pair from the server:

```bash
./ehsm-kms_enroll_app -a http://<your_kms_external_ipaddr>:9000/ehsm/


INFO [main.cpp(45) -> main]: ehsm-kms enroll app start.
INFO [main.cpp(69) -> main]: First handle: send msg0 and get msg1.
INFO [main.cpp(82) -> main]: First handle success.
INFO [main.cpp(84) -> main]: Second handle: send msg2 and get msg3.
INFO [main.cpp(101) -> main]: Second handle success.
INFO [main.cpp(103) -> main]: Third handle: send att_result_msg and get ciphertext of the APP ID and API Key.
appid: b6b6ad56-7741-4d37-9313-3c16754a4f63
apikey: TKLJ9ZqL1gusW7FnGBGh9apk5iJZFVkB
INFO [main.cpp(138) -> main]: decrypt APP ID and API Key success.
INFO [main.cpp(139) -> main]: Third handle success.
INFO [main.cpp(142) -> main]: ehsm-kms enroll app end.
```

### Scheme 3: through RestAPI

```bash
curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"

......

{"code":200,"message":"successful","result":{"apikey":"E8QKpBBapaknprx44FaaTY20rptg54Sg","appid":"8d5dd3b8-3996-40f5-9785-dcb8265981ba"}}
```

## 4. Test BigDL-PCCS-eHSM-KMS with SimpleQuerySparkExample

Test with following scala spark example:
### [SimpleQuerySparkExample](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQuerySparkExample.scala)


## 5. Delete Service from Kubernetes

You can quickly and easily delete BigDL-PCCS-eHSM-KMS from Kubernetes with following commands:

```bash
helm uninstall kms # kms or the other name you specified when starting
kubectl delete pvc couch-persistent-storage-couchdb-0 -n bigdl-pccs-ehsm-kms
```
