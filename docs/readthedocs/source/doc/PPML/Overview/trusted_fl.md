# Trusted FL (Federated Learning)

[Federated Learning](https://en.wikipedia.org/wiki/Federated_learning) is a new tool in PPML (Privacy Preserving Machine Learning), which empowers multi-parities to build united model across different parties without compromising privacy, even if these parities have different datasets or features. In FL training stage, sensitive data will be kept locally, only temp gradients or weights will be safely aggregated by a trusted third-parity. In our design, this trusted third-parity is fully protected by Intel SGX.

A number of FL tools or frameworks have been proposed to enable FL in different areas, i.e., OpenFL, TensorFlow Federated, FATE, Flower and PySyft etc. However, none of them is designed for Big Data scenario. To enable FL in big data ecosystem, BigDL PPML provides a SGX-based End-to-end Trusted FL platform. With this platform, data scientist and developers can easily setup FL applications upon distributed large scale datasets with a few clicks. To achieve this goal, we provides following features:

 * ID & feature align: figure out portions of local data that will participate in training stage
 * Horizontal FL: training across multi-parties with same features and different entities
 * Vertical FL: training across multi-parties with same entries and different features.

To ensure sensitive data are fully protected in training and inference stages, we make sure:

 * Sensitive data and weights are kept local, only temp gradients or weights will be safely aggregated by a trusted third-parity
 * Trusted third-parity, i.e., FL Server, is protected by SGX Enclaves
 * Local training environment is protected by SGX Enclaves (recommended but not enforced)
 * Network communication and Storage (e.g., data and model) protected by encryption and Transport Layer Security (TLS)](https://en.wikipedia.org/wiki/Transport_Layer_Security)

That is, even when the program runs in an untrusted cloud environment, all the data and models are protected (e.g., using encryption) on disk and network, and the compute and memory are also protected using SGX Enclaves.

## Prerequisite

Please ensure SGX is properly enabled, and SGX driver is installed. If not, please refer to the [Install SGX Driver](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html#prerequisite).

### Prepare Keys & Dataset

1. Generate the signing key for SGX Enclaves

   Generate the enclave key using the command below, keep it safely for future remote attestations and to start SGX Enclaves more securely. It will generate a file `enclave-key.pem` in the current working directory, which will be the  enclave key. To store the key elsewhere, modify the output file path.

    ```bash
    cd scripts/
    openssl genrsa -3 -out enclave-key.pem 3072
    cd ..
    ```

    Then modify `ENCLAVE_KEY_PATH` in `deploy_fl_container.sh` with your path to `enclave-key.pem`.

2. Prepare keys for TLS with root permission (test only, need input security password for keys). Please also install JDK/OpenJDK and set the environment path of the java path to get `keytool`.

    ```bash
    cd scripts/
    ./generate-keys.sh
    cd ..
    ```

    When entering the passphrase or password, you could input the same password by yourself; and these passwords could also be used for the next step of generating other passwords. Password should be longer than 6 bits and contain numbers and letters, and one sample password is "3456abcd". These passwords would be used for future remote attestations and to start SGX enclaves more securely. And This script will generate 6 files in `./ppml/scripts/keys` dir (you can replace them with your own TLS keys).

    ```bash
    keystore.jks
    keystore.pkcs12
    server.crt
    server.csr
    server.key
    server.pem
    ```

    If run in container, please modify `KEYS_PATH` to `keys/` you generated in last step in `deploy_fl_container.sh`. This dir will mount to container's `/ppml/trusted-big-data-ml/work/keys`, then modify the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml` with container's absolute path. If not in container, just modify the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml` with your local path. If you don't want to build tls channel with certificate, just delete the `privateKeyFilePath` and `certChainFilePath` in `ppml-conf.yaml`.

3. Prepare dataset for FL training. For demo purposes, we have added a public dataset in [BigDL PPML Demo data](https://github.com/intel-analytics/BigDL/tree/branch-2.0/scala/ppml/demo/data). Please download these data into your local machine. Then modify `DATA_PATH` to `./data` with absolute path in your machine and your local ip in `deploy_fl_container.sh`. The `./data` path will mlount to container's `/ppml/trusted-big-data-ml/work/data`, so if you don't run in container, you need to modify the data path in `runH_VflClient1_2.sh`.

### Prepare Docker Image

Pull image from Dockerhub

```bash
docker pull intelanalytics/bigdl-ppml-trusted-big-data-fl-scala-graphene:0.14.0-SNAPSHOT
```

If Dockerhub is not accessable, you can build docker image from BigDL source code

```bash
cd BigDL/scala && bash make-dist.sh -DskipTests -Pspark_3.x
mv ppml/target/bigdl-ppml-spark_3.1.2-0.14.0-SNAPSHOT-jar-with-dependencies.jar ppml/demo
cd ppml/demo
```

Modify your `http_proxy` in `build-image.sh` then run:

```bash
./build-image.sh
```

## Start FLServer

Running this command will start a docker container and initialize the sgx environment.

```bash
bash deploy_fl_container.sh
sudo docker exec -it flDemo bash
./init.sh
```

In container, run:

```bash
./runFlServer.sh
```

The fl-server will start and listen on 8980 port. Both horizontal fl-demo and vertical fl-demo need two clients. You can change the listening port and client number by editing `BigDL/scala/ppml/demo/ppml-conf.yaml`'s `serverPort` and `clientNum`.  

Note that we skip ID & Feature for simplify demo. In practice, before we start Federated Learning, we need to align ID & Feature, and figure out portions of local data that will participate in later training stage. In horizontal FL, feature align is required to ensure each party is training on the same features. In vertical FL, both ID and feature align are required to ensure each party training on different features of the same record.

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
3. [Federated Learning](https://en.wikipedia.org/wiki/Federated_learning)
4. [TensorFlow Federated](https://www.tensorflow.org/federated)
5. [FATE](https://github.com/FederatedAI/FATE)
6. [PySyft](https://github.com/OpenMined/PySyft)
7. [Federated XGBoost](https://github.com/mc2-project/federated-xgboost)
