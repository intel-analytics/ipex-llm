#### Table of Contents  
[What is BigDL PPML?](#what-is-bigdl-ppml)  
[Why BigDL PPML?](#why-bigdl-ppml)  
[Getting Started with PPML](#getting-started-with-ppml)  
[Develop your own Big Data & AI applications with BigDL PPML](#develop-your-own-big-data--ai-applications-with-bigdl-ppml)


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

<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/177922914-f670111c-e174-40d2-b95a-aafe92485024.png" alt="data lifecycle" width='600px' />
</p>

With BigDL PPML, you can run trusted Big Data & AI applications
- Trusted Spark SQL & Dataframe: you can do trusted big data analytics, such as Spark SQL, Dataframe, Spark MLlib
- Trusted ML: you can run trusted machine learning programs, such as MLlib, XGBoost
- Trusted DL: you can run trusted deep learning programs, such as BigDL, Orca, Nano, DLlib
- Trusted FL (Federated Learning): TODO

## Getting Started with PPML

### 0. Preparation your environment
To get started with BigDL PPML, you should prepare your environment first, including K8s cluster setup, K8s-SGX plugin setup, key/secret preparation, KMS and attestation service setup, BigDL PPML Docker Image preparation. More details in [Prepare Environment](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md).

### 1. Encrypt and Upload Data
Upload encrypted data to be used by your application to the nfs server. More details in [Encrypt Your Data](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/encrypt_and_decrypt.md).

### 2. Prepare Big Data & AI applications
Build standard Big Data & AI applications, here we give several existing examples: [examples](#develop-your-own-big-data--ai-applications-with-bigdl-ppml).

To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#develop-your-own-big-data--ai-applications-with-bigdl-ppml).

### 3. Submit Job
You have two options to submit BigDL PPML jobs: use CLI or use helm chart. More details in [Submit BigDL PPML Job](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/submit_job.md).

### 4. Decrypt and Read Result
When the job is done, you can decrypt and read result of the job. More details in [Decrypt Job Result](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/encrypt_and_decrypt.md).


## Develop your own Big Data & AI applications with BigDL PPML

xxxx
