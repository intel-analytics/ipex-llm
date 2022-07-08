## What is BigDL PPML?

Protecting data privacy and confidentiality is critical in a world where data is everywhere. In recent years, more and more countries have enacted data privacy legislation or are expected to pass comprehensive legislation to protect data privacy, the importance of privacy and data protection is increasingly recognized.

To better protect sensitive data, it’s helpful to think about it in all dimensions of data lifecycle: data at rest, data in transit, and data in use. Data being transferred on a network is “in transit”, data in storage is “at rest”, and data being processed is “in use”.

<p align="center">
  <img src="https://user-images.githubusercontent.com/61072813/177720405-60297d62-d186-4633-8b5f-ff4876cc96d6.png" alt="data lifecycle" width='390px' height='260px'/>
</p>

Encryption technology can provide solid protection for data at rest and data that is in transit. For protecting data in transit, enterprises often choose to encrypt sensitive data prior to moving or use encrypted connections (HTTPS, SSL, TLS, FTPS, etc) to protect the contents of data in transit. For protecting data at rest, enterprises can simply encrypt sensitive files prior to storing them or choose to encrypt the storage drive itself. However, the third state, data in use has always been a weakly protected target. 

There are three emerging solutions seek to reduce the data-in-use attack surface: homomorphic encryption, multi-party computation, and confidential computing. Among these, [Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) protects data in use by performing computation in a hardware-based [Trusted Execution Environment (TEE)](https://en.wikipedia.org/wiki/Trusted_execution_environment). [Intel® SGX](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html) is Intel’s Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. [Intel® TDX](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html) is the next generation Intel’s Trusted Execution Environment (TEE), introducing new, architectural elements to help deploy hardware-isolated, virtual machines (VMs) called trust domains (TDs).

[PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) (Privacy Preserving Machine Learning) in [BigDL 2.0](https://github.com/intel-analytics/BigDL) provides a Trusted Cluster Environment for secure Big Data & AI applications, even on untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning, homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

## Why BigDL PPML?

PPML allows organizations to explore powerful AI techniques while working to minimize the security risks associated with handling large amounts of sensitive data. PPML protects data at rest, in transit and in use: compute and memory protected by SGX Enclaves, storage (e.g., data and model) protected by encryption, network communication protected by remote attestation and Transport Layer Security (TLS), and optional Federated Learning support. 

![image](https://user-images.githubusercontent.com/61072813/177907214-2cc629d7-374b-4b51-8f23-e8514678f032.png)

With BigDL PPML, you can run trusted Big Data & AI applications
- Trusted Spark SQL & Dataframe: you can do trusted big data analytics, such as Spark SQL, Dataframe, Spark MLlib
- Trusted ML: you can run trusted machine learning programs, such as MLlib, XGBoost
- Trusted DL: you can run trusted deep learning programs, such as BigDL, Orca, Nano, DLlib
- Trusted FL (Federated Learning): TODO

## Getting Started with PPML

### 0. Prerequisite
* Set up K8s cluster: placeholder
* Set up K8s-SGX plugin: [deploy_intel_sgx_device_plugin_for_k8s](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html)
* Set up Attestation service: placeholder
* Set up KMS (key management service): [ehsm-kms](https://github.com/intel-analytics/BigDL/blob/main/ppml/services/pccs-ehsm/kubernetes/README.md)
* (Optional) Set up K8s Monitioring: [bigdl-ppml-sgx-k8s-prometheus/README.md](https://github.com/analytics-zoo/ppml-e2e-examples/blob/main/bigdl-ppml-sgx-k8s-prometheus/README.md)
* Prepare BigDL PPML Docker Image

    * Pull Docker image from Dockerhub
        ```
        docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:2.1.0-SNAPSHOT
        ```
    * Alternatively, you can build Docker image from Dockerfile (this will take some time):
        ```
        cd trusted-big-data-ml/python/docker-graphene
        ./build-docker-image.sh
        ```
### 1. Encrypt and Upload Data
* Generate input file
* Encrypt file
* Upload to nfs server

### 2. Build App
* Buid standard Big Data and ML applications (Spark, Flink, XGBoost, Tensorflow, PyTorch, OpenVINO, Ray). Optionally use BigDL PPML APIs (ctypto, VFL, etc.)

### 3. Submit Job
* Start BigDL PPML container
* CLI to submit job to K8s (only demo one mode here, put a video and doc link here to refer all 4 modes)
* Check the the state of driver and executor, whether it runs in sgx
* Check k8s monitioring

### 4. Decrypt and Read Result
* check the output, which is encrypted
* decrypt the output
* show the decrypted content

```
#### This is just a note about videos, will remove this section when videos are ready
We should have the following videos:
1. how to deploy kms and encrypt/decrypt
2. how to deploy attestation and use it
3. how to deploy k8s monitoring
4. unmodified application (PPMLContext)
5. how to submit job (start ppml container, ppml cli introduction, 4 modes)
6. runtime (check the state of driver and executor, whether it runs in sgx, and k8s monitioring)
```

## Run your application on PPML

xxxx
