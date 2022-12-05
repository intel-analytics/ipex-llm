Protecting privacy and confidentiality is critical for large-scale data analysis and machine learning. **BigDL PPML** (BigDL Privacy Preserving Machine Learning) combines various low-level hardware and software security technologies (e.g., Intel® Software Guard Extensions (Intel® SGX), Security Key Management, Remote Attestation, Data Encryption, Federated Learning, etc.) so that users can continue applying standard Big Data and AI technologies (such as Apache Spark, Apache Flink, TensorFlow, PyTorch, etc.) without sacrificing privacy. 

#### Table of Contents  
[1. What is BigDL PPML?](#1-what-is-bigdl-ppml)  
[2. Why BigDL PPML?](#2-why-bigdl-ppml)  
[3. Getting Started with PPML](#3-getting-started-with-ppml)  \
&ensp;&ensp;[3.1 BigDL PPML Hello World](#31-bigdl-ppml-hello-world) \
&ensp;&ensp;[3.2 BigDL PPML End-to-End Workflow](#32-bigdl-ppml-end-to-end-workflow) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 0. Preparation your environment](#step-0-preparation-your-environment): detailed steps in [Prepare Environment](./docs/prepare_environment.md) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 1. Prepare your PPML image for production environment](#step-1-prepare-your-ppml-image-for-production-environment) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 2. Encrypt and Upload Data](#step-2-encrypt-and-upload-data) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 3. Build Big Data & AI applications](#step-3-build-big-data--ai-applications) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 4. Attestation ](#step-4-attestation) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 5. Submit Job](#step-5-submit-job): 4 deploy modes and 2 options to submit job  \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 6. Monitor Job by History Server](#step-6-monitor-job-by-history-server) \
&ensp;&ensp;&ensp;&ensp;&ensp;&ensp;[Step 7. Decrypt and Read Result](#step-7-decrypt-and-read-result) \
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
In this section, you can get started with running a simple native python HelloWorld program and a simple native Spark Pi program locally in a BigDL PPML local docker container to get an initial understanding of the usage of ppml. 

<details><summary>Click to see detailed steps</summary>

**a. Prepare Images**

For demo purpose, we will skip building the custom image here and use the public reference image provided by BigDL PPML `intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT` to have a quick start.

Note: This public image is only for demo purposes, it is non-production. For security concern, you are strongly recommended to generate your encalve key and build your own custom image for your production environment. Refer to [How to Prepare Your PPML image for production environment](#step-1-prepare-your-ppml-image-for-production-environment).

**b. Prepare Keys**

* generate ssl_key

  Download scripts from [here](https://github.com/intel-analytics/BigDL).

  ```
  cd BigDL/ppml/
  sudo bash scripts/generate-keys.sh
  ```
  This script will generate keys under keys/ folder

**c. Start the BigDL PPML Local Container**
```
# KEYS_PATH means the absolute path to the keys folder in step a
# LOCAL_IP means your local IP address.
export KEYS_PATH=YOUR_LOCAL_KEYS_PATH
export LOCAL_IP=YOUR_LOCAL_IP
# ppml graphene image is deprecated, please use the gramine version
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT

sudo docker pull $DOCKER_IMAGE

sudo docker run -itd \
    --privileged \
    --net=host \
    --cpuset-cpus="0-5" \
    --oom-kill-disable \
    --device=/dev/gsgx \
    --device=/dev/sgx/enclave \
    --device=/dev/sgx/provision \
    -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
    -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
    --name=bigdl-ppml-client-local \
    -e LOCAL_IP=$LOCAL_IP \
    -e SGX_MEM_SIZE=64G \
    $DOCKER_IMAGE bash
```

**d. Run Python HelloWorld in BigDL PPML Local Container**

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/base/scripts/start-python-helloword-on-sgx.sh) to run trusted [Python HelloWorld](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/base/examples/helloworld.py) in BigDL PPML client container:
```
sudo docker exec -it bigdl-ppml-client-local bash work/scripts/start-python-helloword-on-sgx.sh
```
Check the log:
```
sudo docker exec -it bigdl-ppml-client-local cat /ppml/trusted-big-data-ml/test-helloworld-sgx.log | egrep "Hello World"
```
The result should look something like this:
> Hello World


**e. Run Spark Pi in BigDL PPML Local Container**

Run the [script](https://github.com/intel-analytics/BigDL/blob/main/ppml/trusted-big-data-ml/python/docker-gramine/base/scripts/start-spark-pi-on-local-sgx.sh) to run trusted [Spark Pi](https://github.com/apache/spark/blob/v3.1.2/examples/src/main/python/pi.py) in BigDL PPML client container:

```bash
sudo docker exec -it bigdl-ppml-client-local bash work/scripts/start-spark-pi-on-local-sgx.sh
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

Next, you are going to build a base image, and a custom image on top of it in order to avoid leaving secrets e.g. enclave key in images/containers. After that, you need to register the mrenclave in your custom image to Attestation Service Before running your application, and PPML will verify the runtime MREnclave automatically at the backend. The below chart illustrated the whole workflow:
![PPML Workflow with MREnclave](https://user-images.githubusercontent.com/60865256/197942436-7e40d40a-3759-49b4-aab1-826f09760ab1.png)

Start your application with the following guide step by step:

#### Step 1. Prepare your PPML image for production environment
To build a secure PPML image which can be used in production environment, BigDL prepared a public base image that does not contain any secrets. You can customize your own image on top of this base image.

1. Prepare BigDL Base Image

    Users can pull the base image from dockerhub or build it by themselves. 

    Pull the base image
    ```bash
    docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-base:2.2.0-SNAPSHOT
    ```
    or

    Running the following command to build the BigDL base image first. Please update the parameters in `./base/build-base-image.sh` first. 

    ```bash
    cd base
    # configure parameters in build-base-image.sh please
    ./build-base-image.sh
    cd ..
    ```
    
2. Build Custom Image

    When the base image is ready, you need to generate your enclave key which will be used when building custom image, keep the enclave key safely for future remote attestations.

    Running the following command to generate the enclave key `enclave-key.pem` , which is used to launch and sign SGX Enclave. 

    ```bash
    cd bigdl-gramine
    openssl genrsa -3 -out enclave-key.pem 3072
    ```

    When the enclave key `enclave-key.pem` is generated, you are ready to build your custom image by running the following command: 

    ```bash
    # under bigdl-gramine dir
    # modify custom parameters in build-custom-image.sh
    ./build-custom-image.sh
    cd ..
    ```
    **Warning:** If you want to skip DCAP attestation in runtime containers, you can set `ENABLE_DCAP_ATTESTATION` to *false* in `build-custom-image.sh`, and this will generate a none-attestation image. **But never do this unsafe operation in producation!**

    The sensitive encalve key will not be saved in the built image. Two values `mr_enclave` and `mr_signer` are recorded while the Enclave is built, you can find `mr_enclave` and `mr_signer` values in the console log, which are hash values and used to register your MREnclave in the following attestation step.

    ````bash
    [INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
    mr_enclave       : c7a8a42af......
    mr_signer        : 6f0627955......
    ````

    Note: you can also customize the image according to your own needs, e.g. install extra python library, add code, jars.
    
    Then, start a client container:

    ```
    export K8S_MASTER=k8s://$(sudo kubectl cluster-info | grep 'https.*6443' -o -m 1)
    echo The k8s master is $K8S_MASTER .
    export DATA_PATH=/YOUR_DIR/data
    export KEYS_PATH=/YOUR_DIR/keys
    export SECURE_PASSWORD_PATH=/YOUR_DIR/password
    export KUBECONFIG_PATH=/YOUR_DIR/kubeconfig
    export LOCAL_IP=$LOCAL_IP
    export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-big-data-ml-python-gramine-reference:2.2.0-SNAPSHOT # or the custom image built by yourself

    sudo docker run -itd \
        --privileged \
        --net=host \
        --name=bigdl-ppml-client-k8s \
        --cpuset-cpus="0-4" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $DATA_PATH:/ppml/trusted-big-data-ml/work/data \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        -v $SECURE_PASSWORD_PATH:/ppml/trusted-big-data-ml/work/password \
        -v $KUBECONFIG_PATH:/root/.kube/config \
        -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
        -e RUNTIME_K8S_SPARK_IMAGE=$DOCKER_IMAGE \
        -e LOCAL_IP=$LOCAL_IP \
        $DOCKER_IMAGE bash
    ```
    

#### Step 2. Encrypt and Upload Data
Encrypt the input data of your Big Data & AI applications (here we use SimpleQuery) and then upload encrypted data to the nfs server. More details in [Encrypt Your Data](./services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

1. Generate the input data `people.csv` for SimpleQuery application
you can use [generate_people_csv.py](https://github.com/intel-analytics/BigDL/tree/main/ppml/scripts/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.

2. Encrypt `people.csv`
    ```
    docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $input_file_path"
    ```
#### Step 3. Build Big Data & AI applications
To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml). The code of SimpleQuery is in [here](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQuerySparkExample.scala), it is already built into bigdl-ppml-spark_${SPARK_VERSION}-${BIGDL_VERSION}.jar, and the jar is put into PPML image.

#### Step 4. Attestation 

   Enter the client container:
   ```
   sudo docker exec -it bigdl-ppml-client-k8s bash
   ```
   
1. Disable attestation

    If you do not need the attestation, you can disable the attestation service. You should configure spark-driver-template.yaml and spark-executor-template.yaml to set `ATTESTATION` value to `false`. By default, the attestation service is disabled. 
    ``` yaml
    apiVersion: v1
    kind: Pod
    spec:
      ...
        env:
          - name: ATTESTATION
            value: false
      ...
    ```

2. Enable attestation

    The bi-attestation gurantees that the MREnclave in runtime containers is a secure one made by you. Its workflow is as below:
    ![image](https://user-images.githubusercontent.com/60865256/198168194-d62322f8-60a3-43d3-84b3-a76b57a58470.png)
    
    To enable attestation, first you should have a running Attestation Service in your environment. 

    **2.1. Deploy EHSM KMS & AS**

      KMS (Key Management Service) and AS (Attestation Service) make sure applications of the customer actually run in the SGX MREnclave signed above by customer-self, rather than a fake one fake by an attacker.

      BigDL PPML use EHSM as reference KMS&AS, you can follow the guide [here](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/ehsm/kubernetes#deploy-bigdl-ehsm-kms-on-kubernetes-with-helm-charts) to deploy EHSM in your environment.

    **2.2. Enroll in EHSM**

    Execute the following command to enroll yourself in EHSM, The `<kms_ip>` is your configured-ip of EHSM service in the deployment section:

    ```bash
    curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"
    ......
    {"code":200,"message":"successful","result":{"apikey":"E8QKpBB******","appid":"8d5dd3b*******"}}
    ```

    You will get a `appid` and `apikey` pair, save it for later use.

    **2.3. Attest EHSM Server (optional)**

    You can attest the EHSM server and verify the service is trusted before running workloads, that avoids sending your secrets to a fake EHSM service.

    To attest EHSM server, first, start a bigdl container using the custom image build before. **Note**: this is the other container different from the client.

    ```bash
    export KEYS_PATH=YOUR_LOCAL_SPARK_SSL_KEYS_FOLDER_PATH
    export LOCAL_IP=YOUR_LOCAL_IP
    export CUSTOM_IMAGE=YOUR_CUSTOM_IMAGE_BUILT_BEFORE
    export PCCS_URL=YOUR_PCCS_URL # format like https://1.2.3.4:xxxx, obtained from KMS services or a self-deployed one

    sudo docker run -itd \
        --privileged \
        --net=host \
        --cpuset-cpus="0-5" \
        --oom-kill-disable \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $KEYS_PATH:/ppml/trusted-big-data-ml/work/keys \
        --name=gramine-verify-worker \
        -e LOCAL_IP=$LOCAL_IP \
        -e PCCS_URL=$PCCS_URL \
        $CUSTOM_IMAGE bash
    ```

    Enter the docker container:

    ```bash
    sudo docker exec -it gramine-verify-worker bash
    ```

    Set the variables in `verify-attestation-service.sh` before running it:

      ```
      `ATTESTATION_URL`: URL of attestation service. Should match the format `<ip_address>:<port>`.

      `APP_ID`, `API_KEY`: The appID and apiKey pair generated by your attestation service.

      `ATTESTATION_TYPE`: Type of attestation service. Currently support `EHSMAttestationService`.

      `CHALLENGE`: Challenge to get quote of attestation service which will be verified by local SGX SDK. Should be a BASE64 string. It can be a casual BASE64 string, for example, it can be generated by the command `echo anystring|base64`.
      ```

    In the container, execute `verify-attestation-service.sh` to verify the attestation service quote.

      ```bash
      bash verify-attestation-service.sh
      ```

    **2.4. Register your MREnclave to EHSM**

    Register the MREnclave with metadata of your MREnclave (appid, apikey, mr_enclave, mr_signer) obtained in above steps to EHSM through running a python script:

    ```bash
    # At /ppml/trusted-big-data-ml inside the container now
    python register-mrenclave.py --appid <your_appid> \
                                --apikey <your_apikey> \
                                --url https://<kms_ip>:9000 \
                                --mr_enclave <your_mrenclave_hash_value> \
                                --mr_signer <your_mrensigner_hash_value>
    ```
    You will receive a response containing a `policyID` and save it which will be used to attest runtime MREnclave when running distributed kubernetes application.

    **2.5. Enable Attestation in configuration**

    First, upload `appid`, `apikey` and `policyID` obtained before to kubernetes as secrets:
    
    ```bash
    kubectl create secret generic kms-secret \
                      --from-literal=app_id=YOUR_KMS_APP_ID \
                      --from-literal=api_key=YOUR_KMS_API_KEY \
                      --from-literal=policy_id=YOUR_POLICY_ID
    ```
    
    Configure `spark-driver-template.yaml` and `spark-executor-template.yaml` to enable Attestation as follows:
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
          - name: PCCS_URL
            value: your_pccs_url  -----> <set_the_value_to_your_pccs_url>
          - name: ATTESTATION_URL
            value: your_attestation_url
          - name: APP_ID
            valueFrom:
              secretKeyRef:
                name: kms-secret
                key: app_id
          - name: API_KEY
            valueFrom:
              secretKeyRef:
                name: kms-secret
                key: app_key
          - name: ATTESTATION_POLICYID
            valueFrom:
              secretKeyRef:
                name: policy-id-secret
                key: policy_id
    ...
    ```
    You should get `Attestation Success!` in logs after you [submit a PPML job](#step-4-submit-job) if the quote generated with user report is verified successfully by Attestation Service, or you will get `Attestation Fail! Application killed!` and the job will be stopped.

#### Step 5. Submit Job
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
              --sgx-driver-jvm-memory 12g \
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


#### Step 6. Monitor Job by History Server
You can monitor spark events using history server. The history server provides an interface to watch and log spark performance and metrics.
     
First, create a shared directory that can be accessed by both the client and the other worker containers in your cluster. For example, you can create an empty directory under the mounted nfs path or hdfs. The spark drivers and executors will write their event logs to this destination, and the history server will read logs here as well.
     
Second, enter your client container and edit `$SPARK_HOME/conf/spark-defaults.conf`, where the histroy server reads the configurations:
```
spark.eventLog.enabled           true
spark.eventLog.dir               <your_shared_dir_path> ---> e.g. file://<your_nfs_dir_path> or hdfs://<your_hdfs_dir_path>
spark.history.fs.logDirectory    <your_shared_dir_path> ---> similiar to spark.eventLog.dir
```
     
Third, run the below command and the history server will start to watch automatically:
```
$SPARK_HOME/sbin/start-history-server.sh
```
     
Next, when you run spark jobs, enable writing driver and executor event logs in java/spark-submit commands by setting spark conf like below:
```
...
--conf spark.eventLog.enabled=true \
--conf spark.eventLog.dir=<your_shared_dir_path> \
...
```
     
Starting spark jobs, you can find event log files at `<your_shared_dir_path>` like:
```
$ ls
local-1666143241860 spark-application-1666144573580
     
$ cat spark-application-1666144573580
......
{"Event":"SparkListenerJobEnd","Job ID":0,"Completion Time":1666144848006,"Job Result":{"Result":"JobSucceeded"}}
{"Event":"SparkListenerApplicationEnd","Timestamp":1666144848021}
```
     
You can use these logs to analyze spark jobs. Moreover, you are also allowed to surf from a web UI provided by the history server by accessing `http://localhost:18080`:
![history server UI](https://user-images.githubusercontent.com/60865256/196840282-6584f36e-5e72-4144-921e-4536d3391f05.png)    


#### Step 7. Decrypt and Read Result
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
             "spark.bigdl.kms.simple.key" -> "your_api_key",
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
                   "simple_api_key": "your_api_key",
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
             "spark.bigdl.kms.ehs.key" -> "your_api_key",
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
                   "ehsm_api_key": "your_api_key",
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
       "spark.bigdl.kms.simple.key" -> "your_api_key",
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
                "simple_api_key": "your_api_key",
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
