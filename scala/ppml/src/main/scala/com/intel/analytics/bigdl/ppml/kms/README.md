# 1. Deploy EHSM KMS on Kubernetes

![Deploy eHSM KMS on Kubernetes](https://user-images.githubusercontent.com/60865256/160524763-59ba22d5-dc93-4755-a993-a488cf48a8f9.png)


## Prerequisites

- Ensure you already have a running kubenetes cluster environment, if not, please follow [k8s-setup-guide](https://github.com/intel/ehsm/blob/main/docs/k8s-setup-guide.md) to setup the K8S cluster.
- Ensure you already have a NFS server, if not, please follow [nfs-setup-guide](https://github.com/intel/ehsm/blob/main/docs/nfs-setup-guide.md) to setup a nfs server.



## Deployment

First, download eHSM and couchdb images needed:

```bash
docker pull intelccc/ehsm_kms:0.2.1 #Please make sure the version number is the latest, 0.2.1 when writing
docker pull intelccc/ehsm_dkeycache:0.2.1
docker pull couchdb:3.2
```

Copy the following and save to a `ehsm-kms.yaml`:

```yaml
target/spark-encrypt-io-0.2-SNAPSHOT-jar-with-dependencies.jar
```

Modify the following parameters in the yaml file:

```yaml
......
data:
    dkeyserver_ip: "1.2.3.4"               --> <your_dkeyserver_ip>
    dkeyserver_port: "8888"                --> <your_dkeyserver_port>
    pccs_url: "https://1.2.3.4:8081"       --> <your_nfs_folder>


nfs:
    path: /nfs_ehsm_db                     --> <your_nfs_folder>
    server: 1.2.3.4                        --> <your_nfs_ip>

 containers:
  - name: dkeycache
    image: intelccc/ehsm_dkeycache:latest     --> <your_dkeycache_image>

initContainers:
   - name: init-ehsm-kms
    image: intelccc/ehsm_kms_service:latest   --> <your_kms_image>

containers:
  - name: ehsm-kms
    image: intelccc/ehsm_kms_service:latest   --> <your_kms_image>

kind: Service
metadata:
name: ehsm-kms-service
namespace: ehsm-kms
....
externalIPs:
- 1.2.3.4                                --> <your_kms_external_ipaddr, you need try to find an unused IP>
```

Create namespace and apply the yaml file on your kubernetes cluster:

```bash
# Create ehsm-kms namespace
$ kubectl create namespace ehsm-kms

# apply the yaml file with ehsm-kms namespace
$ kubectl apply -f ehsm-kms.yaml -n ehsm-kms
```

Check as below:

```bash
$ kubectl get all -n ehsm-kms
NAME READY STATUS RESTARTS AGE
pod/couchdb-0 1/1 Running 0 117s
pod/ehsm-kms-deployment-7cd688cddb-7rjlc 1/1 Running 0 117s
pod/ehsm-kms-deployment-7cd688cddb-rc579 1/1 Running 0 117s
pod/ehsm-kms-deployment-7cd688cddb-sl9kd 1/1 Running 0 117s



NAME TYPE CLUSTER-IP EXTERNAL-IP PORT(S) AGE
service/couchdb ClusterIP None <none> 5984/TCP 117s
service/ehsm-kms-service LoadBalancer 10.101.238.92 172.20.55.90 9000:30000/TCP 117s



NAME READY UP-TO-DATE AVAILABLE AGE
deployment.apps/ehsm-kms-deployment 3/3 3 3 117s



NAME DESIRED CURRENT READY AGE
replicaset.apps/ehsm-kms-deployment-7cd688cddb 3 3 3 117s



NAME READY AGE
statefulset.apps/couchdb 1/1 117s

$ curl http://<KMS_SERVER_ETERNAL_IP>:9000/ehsm/?Action=GetVersion
{"code":200,"message":"success!","result":{"git_sha":"ab60af6","version":"0.2.0"}}

```
![eHSM KMS components](https://user-images.githubusercontent.com/60865256/160728446-c8072388-b442-4e24-ba4e-28c6249112c6.png)



## Problem and Solution:

1. When you check ehsm-kms status, if ehsm-kms-deployment pods keep ***CrashLoopBack***, please make sure that you are using the latest eHSM image rather than the older ones.



# 2. Enroll through ehsm-kms_enroll_app

![KMS Key Management](https://user-images.githubusercontent.com/60865256/160524707-4b9576f3-f239-40a9-a228-9c7fec2d10f5.png)

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
./ehsm-kms_enroll_app http://<your_kms_external_ipaddr>:9000/ehsm/


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

# 3. Start EHSMKeyManagementService with LocalCryptoExample

### [LocalCryptoExample](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/src/main/scala/com/intel/analytics/bigdl/ppml/examples/LocalCryptoExample.scala)

```bash
java -cp target/spark-encrypt-io-0.2-SNAPSHOT-jar-with-dependencies.jar \
  com.intel.analytics.bigdl.ppml.examples.LocalCryptoExample \
  --inputPath /your/single/data/file/to/encrypt/and/decrypt \
  --primaryKeyPath /the/path/you/want/to/put/encrypted/primary/key/at \
  --dataKeyPath /the/path/you/want/to/put/encrypted/data/key/at \
  --kmsServerIP /the/kms/external/ip/prementioned \
  --kmsServerPort 9000 \
  --ehsmAPPID /the/appid/obtained/through/enroll \
  --ehsmAPIKEY /the/apikey/obtained/through/enroll \
  --kmsType EHSMKeyManagementService
```
