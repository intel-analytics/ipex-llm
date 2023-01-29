# PPML End-to-End Workflow Example

## E2E Architecture Overview

In this section we take SimpleQuery as an example to go through the entire BigDL PPML end-to-end workflow. SimpleQuery is simple example to query developers between the ages of 20 and 40 from people.csv.


<p align="center">
  <img src="https://user-images.githubusercontent.com/61072813/178393982-929548b9-1c4e-4809-a628-10fafad69628.png" alt="data lifecycle" />
</p>

<video src="https://user-images.githubusercontent.com/61072813/184758702-4b9809f9-50ac-425e-8def-0ea1c5bf1805.mp4" width="100%" controls></video>

---

## Step 0. Preparation your environment
To secure your Big Data & AI applications in BigDL PPML manner, you should prepare your environment first, including K8s cluster setup, K8s-SGX plugin setup, key/password preparation, key management service (KMS) and attestation service (AS) setup, BigDL PPML client container preparation. **Please follow the detailed steps in** [Prepare Environment](./docs/prepare_environment.md).


## Step 1. Encrypt and Upload Data
Encrypt the input data of your Big Data & AI applications (here we use SimpleQuery) and then upload encrypted data to the nfs server. More details in [Encrypt Your Data](./services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

1. Generate the input data `people.csv` for SimpleQuery application
you can use [generate_people_csv.py](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/spark-encrypt-io/generate_people_csv.py). The usage command of the script is `python generate_people.py </save/path/of/people.csv> <num_lines>`.

2. Encrypt `people.csv`
    ```
    docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh encrypt $appid $apikey $input_file_path"
    ```
## Step 2. Build Big Data & AI applications
To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#4-develop-your-own-big-data--ai-applications-with-bigdl-ppml). The code of SimpleQuery is in [here](https://github.com/intel-analytics/BigDL/blob/main/scala/ppml/src/main/scala/com/intel/analytics/bigdl/ppml/examples/SimpleQuerySparkExample.scala), it is already built into bigdl-ppml-spark_3.1.2-2.1.0-SNAPSHOT.jar, and the jar is put into PPML image.

## Step 3. Attestation

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

## Step 4. Submit Job
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

## Step 5. Decrypt and Read Result
When the job is done, you can decrypt and read result of the job. More details in [Decrypt Job Result](./services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

  ```
  docker exec -i $KMSUTIL_CONTAINER_NAME bash -c "bash /home/entrypoint.sh decrypt $appid $apikey $input_path"
  ```

## Video Demo

<video src="https://user-images.githubusercontent.com/61072813/184758643-821026c3-40e0-4d4c-bcd3-8a516c55fc01.mp4" width="100%" controls></video>
