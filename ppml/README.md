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

For protecting data in transit, enterprises often choose to encrypt sensitive data prior to moving or use encrypted connections (HTTPS, SSL, TLS, FTPS, etc) to protect the contents of data in transit. For protecting data at rest, enterprises can simply encrypt sensitive files prior to storing them or choose to encrypt the storage drive itself. However, the third state, data in use has always been a weakly protected target. There are three emerging solutions seek to reduce the data-in-use attack surface: homomorphic encryption, multi-party computation, and confidential computing. 

Among these, [Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) protects data in use by performing computation in a hardware-based [Trusted Execution Environment (TEE)](https://en.wikipedia.org/wiki/Trusted_execution_environment). [Intel® SGX](https://www.intel.com/content/www/us/en/developer/tools/software-guard-extensions/overview.html) is Intel’s Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. [Intel® TDX](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-trust-domain-extensions.html) is the next generation Intel’s Trusted Execution Environment (TEE), introducing new, architectural elements to help deploy hardware-isolated, virtual machines (VMs) called trust domains (TDs).

[PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) (Privacy Preserving Machine Learning) in [BigDL 2.0](https://github.com/intel-analytics/BigDL) provides a Trusted Cluster Environment for secure Big Data & AI applications, even on untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning, homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

## Why BigDL PPML?
PPML allows organizations to explore powerful AI techniques while working to minimize the security risks associated with handling large amounts of sensitive data. PPML protects data at rest, in transit and in use: compute and memory protected by SGX Enclaves, storage (e.g., data and model) protected by encryption, network communication protected by remote attestation and Transport Layer Security (TLS), and optional Federated Learning support. 

<p align="left">
  <img src="https://user-images.githubusercontent.com/61072813/177922914-f670111c-e174-40d2-b95a-aafe92485024.png" alt="data lifecycle" width='600px' />
</p>

With BigDL PPML, you can run trusted Big Data & AI applications
- **Trusted Spark SQL & Dataframe**: with the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) in a secure and trusted fashion.
- **Trusted ML**: with the trusted Big Data analytics and ML/DL support, users can run distributed machine learning (such as MLlib, XGBoost) in a secure and trusted fashion.
- **Trusted DL**: with the trusted Big Data analytics and ML/DL support, users can run distributed deep learning (such as BigDL, Orca, Nano, DLlib) in a secure and trusted fashion.
- **Trusted FL (Federated Learning)**: TODO

## Getting Started with PPML
In this part, first we will introduce the whole workflow of BigDL PPML，then we will go with examples from simple to hard to use the BigDL PPML workflow.
### End-to-End PPML Workflow
![image](https://user-images.githubusercontent.com/61072813/178393982-929548b9-1c4e-4809-a628-10fafad69628.png)

#### 0. Preparation your environment
To secure your Big Data & AI applications in BigDL PPML manner, you should prepare your environment first, including K8s cluster setup, K8s-SGX plugin setup, data/key/password preparation, KMS and attestation service setup, BigDL PPML Docker Image preparation. More details in [Prepare Environment](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/prepare_environment.md).

#### 1. Encrypt and Upload Data
Encrypt the input data of your Big Data & AI applications and then upload encrypted data to the nfs server. More details in [Encrypt Your Data](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

#### 2. Build Big Data & AI applications
To build your own Big Data & AI applications, refer to [develop your own Big Data & AI applications with BigDL PPML](#develop-your-own-big-data--ai-applications-with-bigdl-ppml).

#### 3. Submit Job
When the Big Data & AI application and its input data is prepared, you are ready to submit BigDL PPML jobs. You have two options to submit jobs: use PPML CLI to run jobs on Kubernetes manually, or use Helm to set everything up automatically. More details in [Submit BigDL PPML Job](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/submit_job.md).

#### 4. Decrypt and Read Result
When the job is done, you can decrypt and read result of the job. More details in [Decrypt Job Result](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/services/kms-utils/docker/README.md#3-enroll-generate-key-encrypt-and-decrypt).

### Examples of using End-to-End PPML Workflow
Here we take SimpleQuery as an example to go through the entire end-to-end PPML workflow.


Find more examples in [PPML examples](https://github.com/liu-shaojun/BigDL/blob/ppml_doc/ppml/docs/examples.md).

## Develop your own Big Data & AI applications with BigDL PPML

xxxx
