#### Table of Contents  
[1. What is BigDL PPML?](#1-what-is-bigdl-ppml)  
[2. Why BigDL PPML?](#2-why-bigdl-ppml)  
[3. Getting Started with PPML](#3-getting-started-with-ppml)  \
&ensp;&ensp;[0. Preparation your environment](): Detailed Steps in  \
&ensp;&ensp;[1. Encrypt and Upload Data]() \
&ensp;&ensp;[2. Build Big Data & AI applications]() \
&ensp;&ensp;[3. Submit Job](): Detailed Steps in  \
&ensp;&ensp;[4. Decrypt and Read Result]() \
[4. Develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml)


## 1. What is BigDL PPML?

Protecting data privacy and confidentiality is critical in a world where data is everywhere. In recent years, more and more countries have enacted data privacy legislation or are expected to pass comprehensive legislation to protect data privacy, the importance of privacy and data protection is increasingly recognized.

To better protect sensitive data, it’s helpful to think about it in all dimensions of data lifecycle: data at rest, data in transit, and data in use. Data being transferred on a network is “in transit”, data in storage is “at rest”, and data being processed is “in use”.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61072813/177720405-60297d62-d186-4633-8b5f-ff4876cc96d6.png" alt="data lifecycle" width='390px' height='260px'/>
</p>

For protecting data in transit, enterprises often choose to encrypt sensitive data prior to moving or use encrypted connections (HTTPS, SSL, TLS, FTPS, etc) to protect the contents of data in transit. For protecting data at rest, enterprises can simply encrypt sensitive files prior to storing them or choose to encrypt the storage drive itself. However, the third state, data in use has always been a weakly protected target. There are three emerging solutions seek to reduce the data-in-use attack surface: homomorphic encryption, multi-party computation, and confidential computing. 

Among these, [Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) protects data in use by performing computation in a hardware-based [Trusted Execution Environment (TEE)](https://en.wikipedia.org/wiki/Trusted_execution_environment). [Intel® SGX](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html) is Intel’s Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. [Intel® TDX](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html) is the next generation Intel’s Trusted Execution Environment (TEE), introducing new, architectural elements to help deploy hardware-isolated, virtual machines (VMs) called trust domains (TDs).

[PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) (Privacy Preserving Machine Learning) in [BigDL 2.0](https://github.com/intel-analytics/BigDL) provides a Trusted Cluster Environment for secure Big Data & AI applications, even on untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning, homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

## 2. Why BigDL PPML?
PPML allows organizations to explore powerful AI techniques while working to minimize the security risks associated with handling large amounts of sensitive data. PPML protects data at rest, in transit and in use: compute and memory protected by SGX Enclaves, storage (e.g., data and model) protected by encryption, network communication protected by remote attestation and Transport Layer Security (TLS), and optional Federated Learning support. 

<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/177922914-f670111c-e174-40d2-b95a-aafe92485024.png" alt="data lifecycle" width='600px' />
</p>

With BigDL PPML, you can run trusted Big Data & AI applications
- **Trusted Spark SQL & Dataframe**: with the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) in a secure and trusted fashion.
- **Trusted ML**: with the trusted Big Data analytics and ML/DL support, users can run distributed machine learning (such as MLlib, XGBoost) in a secure and trusted fashion.
- **Trusted DL**: with the trusted Big Data analytics and ML/DL support, users can run distributed deep learning (such as BigDL, Orca, Nano, DLlib) in a secure and trusted fashion.
- **Trusted FL (Federated Learning)**: TODO

## 3. Getting Started with PPML
In this part, first we use native Python HelloWorld and Spark Pi to verify if the Trusted PPML environment is correctly set up, then we introduce the end-to-end workflow of BigDL PPML and go with an example SimpleQuery to use the BigDL PPML workflow.

### 3.1 BigDL PPML Hello World
In this part, you can started with running a simple native python HelloWorld program and a simple native Spark Pi program in a BigDL PPML client container to get familiar with 

a. Start the BigDL PPML client container
<details><summary>expand/fold to see details</summary>

```
#!/bin/bash

# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" in https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md#prepare-key-and-password
# KEYS_PATH means the absolute path to the keys in https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md#prepare-key-and-password
# LOCAL_IP means your local IP address.
export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:2.1.0-SNAPSHOT

sudo docker pull $DOCKER_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v $ENCLAVE_KEY_PATH:/graphene/Pal/src/host/Linux-SGX/signer/enclave-key.pem \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=spark-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
```

</details>

b. Run Python Helloworld in PPML client Container
    <details><summary>expand/fold to see details</summary>

    Run the script to run Trusted Python Helloworld in PPML container:

        ```
        bash work/start-scripts/start-python-helloworld-sgx.sh
        ```

    Open another terminal and check the log:

    ```bash
    sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-helloworld-sgx.log | egrep "Hello World"
    ```

    The result should look something like this:

    > Hello World

    </details>

c. Trusted Spark Pi in PPML client Container
<details><summary>expand/fold to see details</summary>
Run the script to run trusted Spark Pi in PPML container:

```bash
bash work/start-scripts/start-spark-local-pi-sgx.sh
```

Open another terminal and check the log:

```bash
sudo docker exec -it spark-local cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"
```

The result should look something like this:

> Pi is roughly 3.146760

</details>

### 3.2 End-to-End PPML Workflow
![image](https://user-images.githubusercontent.com/61072813/178393982-929548b9-1c4e-4809-a628-10fafad69628.png)

#### 0. Preparation your environment
To secure your Big Data & AI applications in BigDL PPML manner, you should prepare your environment first, including 
K8s cluster setup
K8s-SGX plugin setup
data/key/password preparation
KMS and attestation service setup
BigDL PPML Client Container preparation
Detailed steps in [Prepare Environment](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md). 


#### 1. Encrypt and Upload Data
Encrypt the input data of your Big Data & AI applications and then upload encrypted data to the nfs server. More details in [Encrypt Your Data](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

#### 2. Build Big Data & AI applications
To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#develop-your-own-big-data--ai-applications-with-bigdl-ppml).

#### 3. Submit Job
When the Big Data & AI application and its input data is prepared, you are ready to submit BigDL PPML jobs. You have two options to submit jobs: use PPML CLI to run jobs on Kubernetes manually, or use Helm to set everything up automatically. More details in [Submit BigDL PPML Job](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/submit_job.md).

#### 4. Decrypt and Read Result
When the job is done, you can decrypt and read result of the job. More details in [Decrypt Job Result](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

### Examples of using End-to-End PPML Workflow
Here we take SimpleQuery as an example to go through the entire end-to-end PPML workflow. SimpleQuery is simple example to query developers between the ages of 20 and 40 from people.csv.

#### 0. Preparation your environment
Setup environment as documented in [Prepare Environment](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md).

#### 1. Encrypt and Upload Data
1. Generate the input data `people.csv` for SimpleQuery application
you can use [generate_people_csv.py](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.

2. Encrypt `people.csv`
    ```
    docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $appkey $input_file_path"
    ```

#### 2. Build Big Data & AI applications
?? Build the application SimpleQuery, move simplequery from e2e repo to bigdl repo??

#### 3. Submit Job
Here we use PPML CLI to run jobs on Kubernetes, here we only demo k8s client mode, check other modes, please see [PPML CLI Usage Examples](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/submit_job.md#usage-examples). Alternatively, you can also use Helm to submit jobs automatically, see the details in [Helm Chart Usage](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/submit_job.md#helm-chart).


1. enter the ppml container
    ```
    docker exec -it ppml-spark-client bash
    ```
2. run simplequery on k8s client mode
    ```
    #!/bin/bash
    export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`
    bash bigdl-ppml-submit.sh \
            --master $RUNTIME_SPARK_MASTER \
            --deploy-mode client \
            --sgx-enabled true \
            --sgx-log-level error \
            --sgx-driver-memory 64g \
            --sgx-driver-jvm-memory 12g \
            --sgx-executor-memory 64g \
            --sgx-executor-jvm-memory 12g \
            --driver-memory 32g \
            --driver-cores 8 \
            --executor-memory 32g \
            --executor-cores 8 \
            --num-executors 2 \
            --conf spark.kubernetes.container.image=$RUNTIME_K8S_SPARK_IMAGE \
            --name spark-pi \
            --verbose \
            --class com.intel.analytics.bigdl.ppml.examples.SimpleQuerySparkExample \
            --jars local:///ppml/trusted-big-data-ml/spark-encrypt-io-0.3.0-SNAPSHOT.jar \
            local:///ppml/trusted-big-data-ml/work/data/simplequery/spark-encrypt-io-0.3.0-SNAPSHOT.jar \
            --inputPath /ppml/trusted-big-data-ml/work/data/simplequery/people_encrypted \
            --outputPath /ppml/trusted-big-data-ml/work/data/simplequery/people_encrypted_output \
            --inputPartitionNum 8 \
            --outputPartitionNum 8 \
            --inputEncryptModeValue AES/CBC/PKCS5Padding \
            --outputEncryptModeValue AES/CBC/PKCS5Padding \
            --primaryKeyPath /ppml/trusted-big-data-ml/work/data/simplequery/keys/primaryKey \
            --dataKeyPath /ppml/trusted-big-data-ml/work/data/simplequery/keys/dataKey \
            --kmsType EHSMKeyManagementService
            --kmsServerIP your_ehsm_kms_server_ip \
            --kmsServerPort your_ehsm_kms_server_port \
            --ehsmAPPID your_ehsm_kms_appid \
            --ehsmAPPKEY your_ehsm_kms_appkey
    ```


3. check runtime status

    Exit the container or open a new terminal

* To check the logs of the Kubernetes job, run
  ```
  sudo kubectl logs $( sudo kubectl get pod | grep spark-pi-job | cut -d " " -f1 )
  ```
* To check the logs of the Spark driver, run
  ```
  sudo kubectl logs $( sudo kubectl get pod | grep "spark-pi-sgx.*-driver" -m 1 | cut -d " " -f1 )
  ```
* To check the logs of an Spark executor, run
  ```
  sudo kubectl logs $( sudo kubectl get pod | grep "spark-pi-.*-exec" -m 1 | cut -d " " -f1 )
  ```


#### 4. Decrypt and Read Result
    
  When job is done, check the result, which shoule be encrypted. Decrypt the result
    
  ```
  docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $appkey $input_path"
  ```

## 4. Develop your own Big Data & AI applications with BigDL PPML

xxxx
