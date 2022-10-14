# PPML Introduction

## 1. What is BigDL PPML?

<video src="https://user-images.githubusercontent.com/61072813/184758908-da01f8ea-8f52-4300-9736-8c5ee981d4c0.mp4" width="100%" controls></video>

---

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
