Protecting privacy and confidentiality is critical for large-scale data analysis and machine learning. **BigDL PPML** (BigDL Privacy Preserving Machine Learning) combines various low-level hardware and software security technologies (e.g., Intel® Software Guard Extensions (Intel® SGX), Security Key Management, Remote Attestation, Data Encryption, Federated Learning, etc.) so that users can continue applying standard Big Data and AI technologies (such as Apache Spark, Apache Flink, TensorFlow, PyTorch, etc.) without sacrificing privacy. 

#### Table of Contents  
[1. What is BigDL PPML?](#1-what-is-bigdl-ppml)  
[2. Why BigDL PPML?](#2-why-bigdl-ppml)  
[3. Getting Started with PPML](#3-getting-started-with-ppml)  \
&ensp;&ensp;[3.1 BigDL PPML Hello World](#31-bigdl-ppml-hello-world) \
&ensp;&ensp;[3.2 BigDL PPML End-to-End Workflow](#32-bigdl-ppml-end-to-end-workflow) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 0. Preparation your environment](#step-0-preparation-your-environment): detailed steps in [Prepare Environment](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 1. Encrypt and Upload Data](#step-1-encrypt-and-upload-data) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 2. Build Big Data & AI applications](#step-2-build-big-data--ai-applications) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 3. Attestation ](#step-3-attestation) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 4. Submit Job](#step-4-submit-job): 4 deploy modes and 2 options to submit job  \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 5. Decrypt and Read Result](#step-5-decrypt-and-read-result) \
&ensp;&ensp;[3.3 More BigDL PPML Examples](#33-more-bigdl-ppml-examples) \
[4. Develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml) \
&ensp;&ensp;[4.1 Create PPMLContext](#41-create-ppmlcontext) \
&ensp;&ensp;[4.2 Read and Write Files](#42-read-and-write-files)



## 1. What is BigDL PPML?



https://user-images.githubusercontent.com/61072813/184758908-da01f8ea-8f52-4300-9736-8c5ee981d4c0.mp4





Protecting data privacy and confidentiality is critical in a world where data is everywhere. In recent years, more and more countries have enacted data privacy legislation or are expected to pass comprehensive legislation to protect data privacy, the importance of privacy and data protection is increasingly recognized.

To better protect sensitive data, it's necessary to ensure security for all dimensions of data lifecycle: data at rest, data in transit, and data in use. Data being transferred on a network is `in transit`, data in storage is `at rest`, and data being processed is `in use`.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61072813/177720405-60297d62-d186-4633-8b5f-ff4876cc96d6.png" alt="data lifecycle" width='390px' height='260px'/>
</p>

To protect data in transit, enterprises often choose to encrypt sensitive data prior to moving or use encrypted connections (HTTPS, SSL, TLS, FTPS, etc) to protect the contents of data in transit. For protecting data at rest, enterprises can simply encrypt sensitive files prior to storing them or choose to encrypt the storage drive itself. However, the third state, data in use has always been a weakly protected target. There are three emerging solutions seek to reduce the data-in-use attack surface: homomorphic encryption, multi-party computation, and confidential computing. 

Among these security technologies, [Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) protects data in use by performing computation in a hardware-based [Trusted Execution Environment (TEE)](https://en.wikipedia.org/wiki/Trusted_execution_environment). [Intel® SGX](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html) is Intel's Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. [Intel® TDX](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html) is the next generation Intel's Trusted Execution Environment (TEE), introducing new, architectural elements to help deploy hardware-isolated, virtual machines (VMs) called trust domains (TDs).

[PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) (Privacy Preserving Machine Learning) in [BigDL 2.0](https://github.com/intel-analytics/BigDL) provides a Trusted Cluster Environment for secure Big Data & AI applications, even on untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning, homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

## 2. Why BigDL PPML?
PPML allows organizations to explore powerful AI techniques while working to minimize the security risks associated with handling large amounts of sensitive data. PPML protects data at rest, in transit and in use: compute and memory protected by SGX Enclaves, storage (e.g., data and model) protected by encryption, network communication protected by remote attestation and Transport Layer Security (TLS), and optional Federated Learning support. 

<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/177922914-f670111c-e174-40d2-b95a-aafe92485024.png" alt="data lifecycle" width='600px' />
</p>

With BigDL PPML, you can run trusted Big Data & AI applications
- **Trusted Spark SQL & Dataframe**: with the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) in a secure and trusted fashion.
- **Trusted ML (Machine Learning)**: with the trusted Big Data analytics and ML/DL support, users can run distributed machine learning (such as MLlib, XGBoost) in a secure and trusted fashion.
- **Trusted DL (Deep Learning)**: with the trusted Big Data analytics and ML/DL support, users can run distributed deep learning (such as BigDL, Orca, Nano, DLlib) in a secure and trusted fashion.
- **Trusted FL (Federated Learning)**: with PSI (Private Set Intersection), Secured Aggregation and trusted federated learning support, users can build united model across different parties without compromising privacy, even if these parities have different datasets or features.

## 3. Getting Started with PPML

### 3.1 BigDL PPML Hello World
In this section, you can get started with running a simple native python HelloWorld program and a simple native Spark Pi program locally in a BigDL PPML client container to get an initial understanding of the usage of ppml. 

<details><summary>Click to see detailed steps</summary>

**a. Prepare Keys**

* generate ssl_key

  Download scripts from [here](https://github.com/intel-analytics/BigDL).

  ```
  cd BigDL/ppml/
  sudo bash scripts/generate-keys.sh
  ```
  This script will generate keys under keys/ folder

* generate enclave-key.pem

  ```
  openssl genrsa -3 -out enclave-key.pem 3072
  ```
  This script generates a file enclave-key.pem which is used to sign image.

**b. Start the BigDL PPML client container**
```
#!/bin/bash

# ENCLAVE_KEY_PATH means the absolute path to the "enclave-key.pem" in step a
# KEYS_PATH means the absolute path to the keys folder in step a
# LOCAL_IP means your local IP address.
export ENCLAVE_KEY_PATH=YOUR_LOCAL_ENCLAVE_KEY_PATH
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-graphene:devel

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
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=bigdl-ppml-client-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
```

**c. Run Python HelloWorld in BigDL PPML Client Container**

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/start-scripts/start-python-helloworld-sgx.sh) to run trusted [Python HelloWorld](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/examples/helloworld.py) in BigDL PPML client container:
```
sudo docker exec -it bigdl-ppml-client-local bash work/start-scripts/start-python-helloworld-sgx.sh
```
Check the log:
```
sudo docker exec -it bigdl-ppml-client-local cat /ppml/trusted-big-data-ml/test-helloworld-sgx.log | egrep "Hello World"
```
The result should look something like this:
> Hello World


**d. Run Spark Pi in BigDL PPML Client Container**

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-graphene/start-scripts/start-spark-local-pi-sgx.sh) to run trusted [Spark Pi](https://github.com/apache/spark/blob/v3.1.2/examples/src/main/python/pi.py) in BigDL PPML client container:

```bash
sudo docker exec -it bigdl-ppml-client-local bash work/start-scripts/start-spark-local-pi-sgx.sh
```

Check the log:

```bash
sudo docker exec -it bigdl-ppml-client-local cat /ppml/trusted-big-data-ml/test-pi-sgx.log | egrep "roughly"
```

The result should look something like this:

> Pi is roughly 3.146760

</details>
<br />

### 3.2 BigDL PPML End-to-End Workflow
![image](https://user-images.githubusercontent.com/61072813/178393982-929548b9-1c4e-4809-a628-10fafad69628.png)
In this section we take SimpleQuery as an example to go through the entire BigDL PPML end-to-end workflow. SimpleQuery is simple example to query developers between the ages of 20 and 40 from people.csv. 




https://user-images.githubusercontent.com/61072813/184758702-4b9809f9-50ac-425e-8def-0ea1c5bf1805.mp4



#### Step 0. Preparation your environment
To secure your Big Data & AI applications in BigDL PPML manner, you should prepare your environment first, including K8s cluster setup, K8s-SGX plugin setup, key/password preparation, key management service (KMS) and attestation service (AS) setup, BigDL PPML client container preparation. **Please follow the detailed steps in** [Prepare Environment](./docs/prepare_environment.md). 


#### Step 1. Encrypt and Upload Data
Encrypt the input data of your Big Data & AI applications (here we use SimpleQuery) and then upload encrypted data to the nfs server. More details in [Encrypt Your Data](./services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

1. Generate the input data `people.csv` for SimpleQuery application
you can use [generate_people_csv.py](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.

2. Encrypt `people.csv`
    ```
    docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $input_file_path"
    ```
#### Step 2. Build Big Data & AI applications
To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml). The code of SimpleQuery is in [here](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQuerySparkExample.scala), it is already built into bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar, and the jar is put into PPML image.

#### Step 3. Attestation 

To enable attestation, you should have a running Attestation Service (EHSM-KMS here for example) in your environment. (You can start a KMS  refering to [this link](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/kms-utils/docker)). Configure your KMS app_id and app_key with `kubectl`, and then configure KMS settings in `spark-driver-template.yaml` and `spark-executor-template.yaml` in the container.
``` bash
kubectl create secret generic kms-secret --from-literal=app_id=your-kms-app-id --from-literal=app_key=your-kms-app-key
```
Configure `spark-driver-template.yaml` for example. (`spark-executor-template.yaml` is similar)
``` yaml
apiVersion: v1
kind: Pod
spec:
  containers:
  - name: spark-driver
    securityContext:
      privileged: true
    env:
      - name: ATTESTATION
        value: true
      - name: ATTESTATION_URL
        value: your_attestation_url
      - name: ATTESTATION_ID
        valueFrom:
          secretKeyRef:
            name: kms-secret
            key: app_id
      - name: ATTESTATION_KEY
        valueFrom:
          secretKeyRef:
            name: kms-secret
            key: app_key
...
```
You should get `Attestation Success!` in logs after you [submit a PPML job](#step-4-submit-job) if the quote generated with user report is verified successfully by Attestation Service, or you will get `Attestation Fail! Application killed!` and the job will be stopped.

#### Step 4. Submit Job
When the Big Data & AI application and its input data is prepared, you are ready to submit BigDL PPML jobs. You need to choose the deploy mode and the way to submit job first.

* **There are 4 modes to submit job**:

    1. **local mode**: run jobs locally without connecting to cluster. It is exactly same as using spark-submit to run your application: `$SPARK_HOME/bin/spark-submit --class "SimpleApp" --master local[4] target.jar`, driver and executors are not protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703141-63209559-05e1-4c4d-b096-6b862a9bed8a.png" width='250px' />
        </p>


    2. **local SGX mode**: run jobs locally with SGX guarded. As the picture shows, the client JVM is running in a SGX Enclave so that driver and executors can be protected.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703165-2afc280d-6a3d-431d-9856-dd5b3659214a.png" width='250px' />
        </p>


    3. **client SGX mode**: run jobs in k8s client mode with SGX guarded. As we know, in K8s client mode, the driver is deployed locally as an external client to the cluster. With **client SGX mode**, the executors running in K8S cluster are protected by SGX, the driver running in client is also protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703216-70588315-7479-4b6c-9133-095104efc07d.png" width='500px' />
        </p>


    4. **cluster SGX mode**: run jobs in k8s cluster mode with SGX guarded. As we know, in K8s cluster mode, the driver is deployed on the k8s worker nodes like executors. With **cluster SGX mode**, the driver and  executors running in K8S cluster are protected by SGX.
        <p align="left">
          <img src="https://user-images.githubusercontent.com/61072813/174703234-e45b8fe5-9c61-4d17-93ef-6b0c961a2f95.png" width='500px' />
        </p>


* **There are two options to submit PPML jobs**:
    * use [PPML CLI](./docs/submit_job.md#ppml-cli) to submit jobs manually
    * use [helm chart](./docs/submit_job.md#helm-chart) to submit jobs automatically

Here we use **k8s client mode** and **PPML CLI** to run SimpleQuery. Check other modes, please see [PPML CLI Usage Examples](./docs/submit_job.md#usage-examples). Alternatively, you can also use Helm to submit jobs automatically, see the details in [Helm Chart Usage](./docs/submit_job.md#helm-chart).

  <details><summary>expand to see details of submitting SimpleQuery</summary>

  1. enter the ppml container
      ```
      docker exec -it bigdl-ppml-client-k8s bash
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
              --name simplequery \
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
              --ehsmAPIKEY your_ehsm_kms_apikey
      ```


  3. check runtime status: exit the container or open a new terminal

      To check the logs of the Spark driver, run
      ```
      sudo kubectl logs $( sudo kubectl get pod | grep "simplequery.*-driver" -m 1 | cut -d " " -f1 )
      ```
      To check the logs of an Spark executor, run
      ```
      sudo kubectl logs $( sudo kubectl get pod | grep "simplequery-.*-exec" -m 1 | cut -d " " -f1 )
      ```
  
  4. If you setup [PPML Monitoring](docs/prepare_environment.md#optional-k8s-monitioring-setup), you can check PPML Dashboard to monitor the status in http://kubernetes_master_url:3000

    ![image](https://user-images.githubusercontent.com/61072813/179948818-a2f6844f-0009-49d1-aeac-2e8c5a7ef677.png)

  </details>
<br />

#### Step 5. Decrypt and Read Result
When the job is done, you can decrypt and read result of the job. More details in [Decrypt Job Result](./services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

  ```
  docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $apikey $input_path"
  ```

https://user-images.githubusercontent.com/61072813/184758643-821026c3-40e0-4d4c-bcd3-8a516c55fc01.mp4



### 3.3 More BigDL PPML Examples
In addition to the above Spark Pi and Python HelloWorld programs running locally, and simplequery application running on the k8s cluster, we also provide other examples including Trusted Data Analysis, Trusted ML, Trusted DL and Trusted FL. You can find these examples in [more examples](./docs/examples.md). 

## 4. Develop your own Big Data & AI applications with BigDL PPML

First you need to create a `PPMLContext`, which wraps `SparkSession` and provides methods to read encrypted data file into plain-text RDD/DataFrame and write DataFrame to encrypted data file. Then you can read & write data through `PPMLContext`.

If you are familiar with Spark, you may find that the usage of `PPMLConext` is very similar to Spark.

### 4.1 Create PPMLContext

- create a PPMLContext with `appName`

   This is the simplest way to create a `PPMLContext`. When you don't need to read/write encrypted files, you can use this way to create a `PPMLContext`.

   <details open>
    <summary>scala</summary>

   ```scala
   import com.intel.analytics.bigdl.ppml.PPMLContext
   
   val sc = PPMLContext.initPPMLContext("MyApp")
   ```

   </details>

  <details>
    <summary>python</summary>

   ```python
   from bigdl.ppml.ppml_context import *
   
   sc = PPMLContext("MyApp")
   ```

   </details>

   If you want to read/write encrypted files, then you need to provide more information.

- create a PPMLContext with `appName` & `ppmlArgs`

   `ppmlArgs` is ppml arguments in a Map, `ppmlArgs` varies according to the kind of Key Management Service (KMS) you are using. Key Management Service (KMS) is used to generate `primaryKey` and `dataKey` to encrypt/decrypt data. We provide 3 types of KMS ——SimpleKeyManagementService, EHSMKeyManagementService, AzureKeyManagementService.

   Refer to [KMS Utils](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/kms-utils/docker/README.md) to use KMS to generate `primaryKey` and `dataKey`, then you are ready to create **PPMLContext** with `ppmlArgs`.

  - For `SimpleKeyManagementService`:

      <details open>
       <summary>scala</summary>
  
      ```scala
      import com.intel.analytics.bigdl.ppml.PPMLContext
      
      val ppmlArgs: Map[String, String] = Map(
             "spark.bigdl.kms.type" -> "SimpleKeyManagementService",
             "spark.bigdl.kms.simple.id" -> "your_app_id",
             "spark.bigdl.kms.simple.key" -> "your_app_key",
             "spark.bigdl.kms.key.primary" -> "/your/primary/key/path/primaryKey",
             "spark.bigdl.kms.key.data" -> "/your/data/key/path/dataKey"
         )
    
      val sc = PPMLContext.initPPMLContext("MyApp", ppmlArgs)
      ```

      </details>
  
  
      <details>
       <summary>python</summary>
  
      ```python
      from bigdl.ppml.ppml_context import *

      ppml_args = {"kms_type": "SimpleKeyManagementService",
                   "simple_app_id": "your_app_id",
                   "simple_app_key": "your_app_key",
                   "primary_key_path": "/your/primary/key/path/primaryKey",
                   "data_key_path": "/your/data/key/path/dataKey"
                  }

      sc = PPMLContext("MyApp", ppml_args)
      ```
      
      </details>

   - For `EHSMKeyManagementService`:

      <details open>
       <summary>scala</summary>
      
      ```scala
      import com.intel.analytics.bigdl.ppml.PPMLContext
         
      val ppmlArgs: Map[String, String] = Map(
             "spark.bigdl.kms.type" -> "EHSMKeyManagementService",
             "spark.bigdl.kms.ehs.ip" -> "your_server_ip",
             "spark.bigdl.kms.ehs.port" -> "your_server_port",
             "spark.bigdl.kms.ehs.id" -> "your_app_id",
             "spark.bigdl.kms.ehs.key" -> "your_app_key",
             "spark.bigdl.kms.key.primary" -> "/your/primary/key/path/primaryKey",
             "spark.bigdl.kms.key.data" -> "/your/data/key/path/dataKey"
      )
         
      val sc = PPMLContext.initPPMLContext("MyApp", ppmlArgs)
      ```
   
     </details>
   
     <details>
       <summary>python</summary>
   
      ```python
      from bigdl.ppml.ppml_context import *
   
      ppml_args = {"kms_type": "EHSMKeyManagementService",
                   "kms_server_ip": "your_server_ip",
                   "kms_server_port": "your_server_port"
                   "ehsm_app_id": "your_app_id",
                   "ehsm_app_key": "your_app_key",
                   "primary_key_path": "/your/primary/key/path/primaryKey",
                   "data_key_path": "/your/data/key/path/dataKey"
                  }
   
      sc = PPMLContext("MyApp", ppml_args)
      ```
   
      </details>

   - For `AzureKeyManagementService`

   
     the parameter `clientId` is not necessary, you don't have to provide this parameter.

      <details open>
       <summary>scala</summary>
      
      ```scala
      import com.intel.analytics.bigdl.ppml.PPMLContext
         
      val ppmlArgs: Map[String, String] = Map(
             "spark.bigdl.kms.type" -> "AzureKeyManagementService",
             "spark.bigdl.kms.azure.vault" -> "key_vault_name",
             "spark.bigdl.kms.azure.clientId" -> "client_id",
             "spark.bigdl.kms.key.primary" -> "/your/primary/key/path/primaryKey",
             "spark.bigdl.kms.key.data" -> "/your/data/key/path/dataKey"
         )
         
      val sc = PPMLContext.initPPMLContext("MyApp", ppmlArgs)
      ```
   
     </details>

     <details>
       <summary>python</summary>
   
       ```python
       from bigdl.ppml.ppml_context import *
   
       ppml_args = {"kms_type": "AzureKeyManagementService",
                    "azure_vault": "your_azure_vault",
                    "azure_client_id": "your_azure_client_id",
                    "primary_key_path": "/your/primary/key/path/primaryKey",
                    "data_key_path": "/your/data/key/path/dataKey"
                   }
   
       sc = PPMLContext("MyApp", ppml_args)
       ```
   
     </details>

- create a PPMLContext with `sparkConf` & `appName` & `ppmlArgs`

   If you need to set Spark configurations, you can provide a `SparkConf` with Spark configurations to create a `PPMLContext`.

   <details open>
    <summary>scala</summary>

   ```scala
   import com.intel.analytics.bigdl.ppml.PPMLContext
   import org.apache.spark.SparkConf
   
   val ppmlArgs: Map[String, String] = Map(
       "spark.bigdl.kms.type" -> "SimpleKeyManagementService",
       "spark.bigdl.kms.simple.id" -> "your_app_id",
       "spark.bigdl.kms.simple.key" -> "your_app_key",
       "spark.bigdl.kms.key.primary" -> "/your/primary/key/path/primaryKey",
       "spark.bigdl.kms.key.data" -> "/your/data/key/path/dataKey"
   )
   
   val conf: SparkConf = new SparkConf().setMaster("local[4]")
   
   val sc = PPMLContext.initPPMLContext(conf, "MyApp", ppmlArgs)
   ```

  </details>

  <details>
    <summary>python</summary>

   ```python
   from bigdl.ppml.ppml_context import *
   from pyspark import SparkConf
   
   ppml_args = {"kms_type": "SimpleKeyManagementService",
                "simple_app_id": "your_app_id",
                "simple_app_key": "your_app_key",
                "primary_key_path": "/your/primary/key/path/primaryKey",
                "data_key_path": "/your/data/key/path/dataKey"
               }
   
   conf = SparkConf()
   conf.setMaster("local[4]")

   sc = PPMLContext("MyApp", ppml_args, conf)
   ```

  </details>

### 4.2 Read and Write Files

To read/write data, you should set the `CryptoMode`:

- `plain_text`: no encryption
- `AES/CBC/PKCS5Padding`: for CSV, JSON and text file
- `AES_GCM_V1`: for PARQUET only
- `AES_GCM_CTR_V1`: for PARQUET only

To write data, you should set the `write` mode:

- `overwrite`: Overwrite existing data with the content of dataframe.
- `append`: Append content of the dataframe to existing data or table.
- `ignore`: Ignore current write operation if data / table already exists without any error.
- `error`: Throw an exception if data or table already exists.
- `errorifexists`: Throw an exception if data or table already exists.

<details open>
  <summary>scala</summary>

```scala
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, PLAIN_TEXT}

// read data
val df = sc.read(cryptoMode = PLAIN_TEXT)
         ...

// write data
sc.write(dataFrame = df, cryptoMode = AES_CBC_PKCS5PADDING)
.mode("overwrite")
...
```

</details>

<details>
  <summary>python</summary>

```python
from bigdl.ppml.ppml_context import *

# read data
df = sc.read(crypto_mode = CryptoMode.PLAIN_TEXT)
  ...

# write data
sc.write(dataframe = df, crypto_mode = CryptoMode.AES_CBC_PKCS5PADDING)
.mode("overwrite")
...
```

</details>

<details><summary>expand to see the examples of reading/writing CSV, PARQUET, JSON and text file</summary>

The following examples use `sc` to represent a initialized `PPMLContext`

**read/write CSV file**

<details open>
  <summary>scala</summary>

```scala
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, PLAIN_TEXT}

// read a plain csv file and return a DataFrame
val plainCsvPath = "/plain/csv/path"
val df1 = sc.read(cryptoMode = PLAIN_TEXT).option("header", "true").csv(plainCsvPath)

// write a DataFrame as a plain csv file
val plainOutputPath = "/plain/output/path"
sc.write(df1, PLAIN_TEXT)
.mode("overwrite")
.option("header", "true")
.csv(plainOutputPath)

// read a encrypted csv file and return a DataFrame
val encryptedCsvPath = "/encrypted/csv/path"
val df2 = sc.read(cryptoMode = AES_CBC_PKCS5PADDING).option("header", "true").csv(encryptedCsvPath)

// write a DataFrame as a encrypted csv file
val encryptedOutputPath = "/encrypted/output/path"
sc.write(df2, AES_CBC_PKCS5PADDING)
.mode("overwrite")
.option("header", "true")
.csv(encryptedOutputPath)
```

</details>

<details>
  <summary>python</summary>

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain csv file and return a DataFrame
plain_csv_path = "/plain/csv/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).option("header", "true").csv(plain_csv_path)

# write a DataFrame as a plain csv file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.option("header", True)
.csv(plain_output_path)

# read a encrypted csv file and return a DataFrame
encrypted_csv_path = "/encrypted/csv/path"
df2 = sc.read(CryptoMode.AES_CBC_PKCS5PADDING).option("header", "true").csv(encrypted_csv_path)

# write a DataFrame as a encrypted csv file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_CBC_PKCS5PADDING)
.mode('overwrite')
.option("header", True)
.csv(encrypted_output_path)
```

</details>

**read/write PARQUET file**

<details open>
  <summary>scala</summary>

```scala
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_GCM_CTR_V1, PLAIN_TEXT}

// read a plain parquet file and return a DataFrame
val plainParquetPath = "/plain/parquet/path"
val df1 = sc.read(PLAIN_TEXT).parquet(plainParquetPath)

// write a DataFrame as a plain parquet file
plainOutputPath = "/plain/output/path"
sc.write(df1, PLAIN_TEXT)
.mode("overwrite")
.parquet(plainOutputPath)

// read a encrypted parquet file and return a DataFrame
val encryptedParquetPath = "/encrypted/parquet/path"
val df2 = sc.read(AES_GCM_CTR_V1).parquet(encryptedParquetPath)

// write a DataFrame as a encrypted parquet file
val encryptedOutputPath = "/encrypted/output/path"
sc.write(df2, AES_GCM_CTR_V1)
.mode("overwrite")
.parquet(encryptedOutputPath)
```

</details>


<details>
  <summary>python</summary>

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain parquet file and return a DataFrame
plain_parquet_path = "/plain/parquet/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).parquet(plain_parquet_path)

# write a DataFrame as a plain parquet file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.parquet(plain_output_path)

# read a encrypted parquet file and return a DataFrame
encrypted_parquet_path = "/encrypted/parquet/path"
df2 = sc.read(CryptoMode.AES_GCM_CTR_V1).parquet(encrypted_parquet_path)

# write a DataFrame as a encrypted parquet file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_GCM_CTR_V1)
.mode('overwrite')
.parquet(encrypted_output_path)
```

</details>

**read/write JSON file**

<details open>
  <summary>scala</summary>

```scala
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, PLAIN_TEXT}

// read a plain json file and return a DataFrame
val plainJsonPath = "/plain/json/path"
val df1 = sc.read(PLAIN_TEXT).json(plainJsonPath)

// write a DataFrame as a plain json file
val plainOutputPath = "/plain/output/path"
sc.write(df1, PLAIN_TEXT)
.mode("overwrite")
.json(plainOutputPath)

// read a encrypted json file and return a DataFrame
val encryptedJsonPath = "/encrypted/parquet/path"
val df2 = sc.read(AES_CBC_PKCS5PADDING).json(encryptedJsonPath)

// write a DataFrame as a encrypted parquet file
val encryptedOutputPath = "/encrypted/output/path"
sc.write(df2, AES_CBC_PKCS5PADDING)
.mode("overwrite")
.json(encryptedOutputPath)
```

</details>

<details>
  <summary>python</summary>

```python
# import
from bigdl.ppml.ppml_context import *

# read a plain json file and return a DataFrame
plain_json_path = "/plain/json/path"
df1 = sc.read(CryptoMode.PLAIN_TEXT).json(plain_json_path)

# write a DataFrame as a plain json file
plain_output_path = "/plain/output/path"
sc.write(df1, CryptoMode.PLAIN_TEXT)
.mode('overwrite')
.json(plain_output_path)

# read a encrypted json file and return a DataFrame
encrypted_json_path = "/encrypted/parquet/path"
df2 = sc.read(CryptoMode.AES_CBC_PKCS5PADDING).json(encrypted_json_path)

# write a DataFrame as a encrypted parquet file
encrypted_output_path = "/encrypted/output/path"
sc.write(df2, CryptoMode.AES_CBC_PKCS5PADDING)
.mode('overwrite')
.json(encrypted_output_path)
```

</details>

**read textfile**

<details open>
  <summary>scala</summary>

```scala
import com.intel.analytics.bigdl.ppml.PPMLContext
import com.intel.analytics.bigdl.ppml.crypto.{AES_CBC_PKCS5PADDING, PLAIN_TEXT}

// read from a plain csv file and return a RDD
val plainCsvPath = "/plain/csv/path"
val rdd1 = sc.textfile(plainCsvPath) // the default cryptoMode is PLAIN_TEXT

// read from a encrypted csv file and return a RDD
val encryptedCsvPath = "/encrypted/csv/path"
val rdd2 = sc.textfile(path=encryptedCsvPath, cryptoMode=AES_CBC_PKCS5PADDING)
```

</details>

<details>
  <summary>python</summary>

```python
# import
from bigdl.ppml.ppml_context import *

# read from a plain csv file and return a RDD
plain_csv_path = "/plain/csv/path"
rdd1 = sc.textfile(plain_csv_path) # the default crypto_mode is "plain_text"

# read from a encrypted csv file and return a RDD
encrypted_csv_path = "/encrypted/csv/path"
rdd2 = sc.textfile(path=encrypted_csv_path, crypto_mode=CryptoMode.AES_CBC_PKCS5PADDING)
```

</details>

</details>

More usage with `PPMLContext` Python API, please refer to [PPMLContext Python API](https://github.com/intel-analytics/BigDL/blob/main/python/ppml/src/bigdl/ppml/README.md).
