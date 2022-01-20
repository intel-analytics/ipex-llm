# Trusted FL (Federated Learning)

SGX-based End-to-end Trusted FL platform

## ID & Feature align

Before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage.

Let RID1 and RID2 be randomized ID from party 1 and party 2.

## Vertical FL

Vertical FL training across multi-parties with different features.

Key features:

* FL Server in SGX
    * ID & feature align
    * Forward & backward aggregation
* Training node in SGX

## Horizontal FL

Horizontal FL training across multi-parties.

Key features:

* FL Server in SGX
   * ID & feature align (optional)
   * Weight/Gradient Aggregation in SGX
* Training Worker in SGX
## Example 

### Before running code

#### **Prepare Docker Image**

##### **Build jar from Source**

```bash
cd BigDL/scala/ppml && mvn clean package -DskipTests -Pspark_3.x
mv target/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar demo
cd demo
```

##### **Build Image**
Modify your `http_proxy` in `build-image.sh` then run:

```bash
./build-image.sh
```

#### **Prepare the Key**

The ppml in bigdl needs secured keys to enable spark security such as Authentication, RPC Encryption, Local Storage Encryption and TLS, you need to prepare the secure keys and keystores. In this tutorial, you can generate keys and keystores with root permission (test only, need input security password for keys).

```bash
bash ../../../ppml/scripts/generate-keys.sh
```

You also need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
openssl genrsa -3 -out enclave-key.pem 3072
```

#### **Prepare the Password**

Next, you need to store the password you used for key generation, i.e., `generate-keys.sh`, in a secured file.

```bash
bash ../../../ppml/scripts/generate-password.sh used_password_when_generate_keys
```

Then modify `ENCLAVE_KEY_PATH` to `enclave-key.pem`, `DATA_PATH` to `BigDL/scala/ppml/demo/data`(for example), `KEYS_PATH` to `your-generated-keys` and `LOCAL_IP` in `deploy_fl_container.sh`.

### **Start container**
Running this command will start a docker container and initialize the sgx environment.

```bash
bash deploy_fl_container.sh
sudo docker exec -it flDemo bash
./init.sh
```

### **Start FLServer**
In container, run:

```bash
./runFlServer.sh
```
The fl-server will start and listen on 8980 port. Both horizontal fl-demo and vertical fl-demo need two clients. You can change the listening port and client number by editing `BigDL/scala/ppml/demo/ppml-conf.yaml`'s `serverPort` and `clientNum`.  

### **HFL Logistic Regression**
Open two new terminals, run:

```bash
sudo docker exec -it flDemo bash
```

to enter the container, then in a terminal run:

```bash
./runHflClient1.sh
```

in another terminal run:

```bash
./runHflClient2.sh
```

Then we start two horizontal fl-clients to cooperate in training a model.

### **VFL Logistic Regression**
Open two new windows, run:

```bash
sudo docker exec -it flDemo bash
```

to enter the container, then in a terminal run:

```bash
./runVflClient1.sh
```

in another terminal run:

```bash
./runVflClient2.sh
```

Then we start two vertical fl-clients to cooperate in training a model.

## References

1. [Intel SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
2. Qiang Yang, Yang Liu, Tianjian Chen, and Yongxin Tong. 2019. Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol. 10, 2, Article 12 (February 2019), 19 pages. DOI:https://doi.org/10.1145/3298981