# Trusted FL (Federated Learning)

Federated Learning is a new tool in PPML (Privacy Preserving Machine Learning), which empowers multi-parities to build united model across different parties without compromising privacy, even if these parities have different datasets or features. In FL training stage, sensitive data will be kept locally, only temp gradients or weights will be safely aggregated by a trusted third-parity. In our design, this trusted third-parity is fully protected by Intel SGX.

A number of FL tools or frameworks have been proposed to enable FL in different areas, i.e., OpenFL, FATE, Flower and PySyft etc. However, none of them is designed for Big Data scenario. To enable FL in big data ecosystem, BigDL PPML provides a SGX-based End-to-end Trusted FL platform. With this platform, data scientist and developers can easily setup FL applications upon distributed large scale datasets with a few clicks. To achieve this goal, we provides following features:

 * ID & feature align: figure out portions of local data that will participate in training stage
 * Horizontal FL: training across multi-parties with same features and different entities
 * Vertical FL: training across multi-parties with same entries and different features.

To ensure sensitive data are fully protected in training and inference stages, we make sure:

 * Sensitive data and weights are kept local, only temp gradients or weights will be safely aggregated by a trusted third-parity
 * Trusted third-parity, i.e., FL Server, is protected by SGX Enclaves
 * Local Training env is protected by SGX Enclaves (recommended but not enforced)
 * Network communication and Storage (e.g., data and model) protected by encryption and Transport Layer Security (TLS)](https://en.wikipedia.org/wiki/Transport_Layer_Security)
 

## Prerequisite

### Prepare Docker Image



```bash
cd BigDL/scala && bash make-dist.sh -DskipTests -Pspark_3.x
mv ppml/target/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar ppml/demo
cd ppml/demo
```

##### Build Image
Modify your `http_proxy` in `build-image.sh` then run:

```bash
./build-image.sh
```

#### **Enclave key**
You need to generate your enclave key using the command below, and keep it safely for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in your present working directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
openssl genrsa -3 -out enclave-key.pem 3072
```

Then modify `ENCLAVE_KEY_PATH` in `deploy_fl_container.sh` with your path to `enclave-key.pem`.

#### **Tls certificate**
If you want to build tls channel with certifacate, you need to prepare the secure keys. In this tutorial, you can generate keys with root permission (test only, need input security password for keys).

**Note: Must enter `localhost` in step `Common Name` for test purpose.**

```bash
sudo bash ../../../ppml/scripts/generate-keys.sh
```

If run in container, please modify `KEYS_PATH` to `keys/` you generated in last step in `deploy_fl_container.sh`. This dir will mount to container's `/ppml/trusted-big-data-ml/work/keys`, then modify the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml` with container's absolute path.

If not in container, just modify the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml` with your local path.

If you don't want to build tls channel with cerfiticate, just delete the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml`.

Then modify `DATA_PATH` to `./data` with absolute path in your machine and your local ip in `deploy_fl_container.sh`. The `./data` path will mlount to container's `/ppml/trusted-big-data-ml/work/data`, so if you don't run in container, you need to modify the data path in `runH_VflClient1_2.sh`.

#### Prepare Docker Image

### Start container
Running this command will start a docker container and initialize the sgx environment.

```bash
bash deploy_fl_container.sh
sudo docker exec -it flDemo bash
./init.sh
```

### Start FLServer
In container, run:

```bash
./runFlServer.sh
```

The fl-server will start and listen on 8980 port. Both horizontal fl-demo and vertical fl-demo need two clients. You can change the listening port and client number by editing `BigDL/scala/ppml/demo/ppml-conf.yaml`'s `serverPort` and `clientNum`.  



## ID & Feature align

Before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage. In horizontal FL, feature align is required to ensure each party is training on the same features. In vertical FL, both ID and feature align are required to ensure each party training on different features of the same record.

Let RID1 and RID2 be randomized ID from party 1 and party 2.


## HFL Logistic Regression

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

## VFL Logistic Regression
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
