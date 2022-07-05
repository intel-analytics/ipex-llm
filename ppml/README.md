## What is BigDL PPML

### Importance of Data Protection
Protecting data privacy and confidentiality is critical in a world where we are constantly storing, consuming, and sharing sensitive data. With the rapid development of information technology, more and more countries have enacted data privacy legislation or are expected to pass comprehensive legislation to protect data privacy, the importance of privacy and data protection is increasingly recognized.

To better protect sensitive data, it’s helpful to think about it in all dimensions of data lifecycle: data at rest, data in transit, and data in use. Data being transferred on a network is “in transit”, data in storage is “at rest”, and data being processed is “in use”. 

<img width="430" alt="image" src="https://user-images.githubusercontent.com/61072813/177228389-f4bba090-d7b1-4413-bab6-c98f2217dac8.png">

Encryption technology can provide solid protection for data at rest and data that is in transit. For protecting data in transit, enterprises often choose to encrypt sensitive data prior to moving or use encrypted connections (HTTPS, SSL, TLS, FTPS, etc) to protect the contents of data in transit. For protecting data at rest, enterprises can simply encrypt sensitive files prior to storing them or choose to encrypt the storage drive itself. However, the third state, when data is in use, is overlooked because of inadequate safeguard mechanisms. With several severe malware attacks have happened at the in-use state, securing data in use has become critical.


### Confidential Computing
[Confidential computing](https://www.intel.com/content/www/us/en/security/confidential-computing.html) is an emerging industry initiative focused on helping to secure data in use. Confidential Computing is the protection of data in use by performing computation in a hardware-based Trusted Execution Environment(TEE).

The efforts can enable encrypted data to be processed in memory while lowering the risk of exposing it to the rest of the system, thereby reducing the potential for sensitive data to be exposed while providing a higher degree of control and transparency for users. In multi-tenant cloud environments, where sensitive data is meant to be kept isolated from other privileged portions of the system stack, Intel® Software Guard Extensions (Intel® SGX) plays a large role in making this capability a reality.

As computing moves to span multiple environments—from on-prem to public cloud to edge—organizations need protection controls that help safeguard sensitive IP and workload data wherever the data resides.

### BigDL PPML

[BigDL PPML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html) provides a Trusted Cluster Environment for secure Big Data & AI applications, even on untrusted cloud environment. By combining Intel Software Guard Extensions (SGX) with several other security technologies (e.g., attestation, key management service, private set intersection, federated learning,  homomorphic encryption, etc.), BigDL PPML ensures end-to-end security enabled for the entire distributed workflows, such as Apache Spark, Apache Flink, XGBoost, TensorFlow, PyTorch, etc.

BigDL provides a distributed PPML platform for protecting the end-to-end Big Data AI pipeline (from data ingestion, data analysis, all the way to machine learning and deep learning):
- Compute and memory protected by SGX Enclaves
- Network communication protected by remote attestation and Transport Layer Security (TLS)
- Storage (e.g., data and model) protected by encryption
- Optional Federated Learning support


## What can BigDL PPML do
BigDL PPML allows users to run unmodified Big Data analysis (such as Spark SQL, Dataframe, Spark MLlib, etc.) and ML/DL programs (such as Apache Spark, Apache Flink, Tensorflow, PyTorch, etc.) in a secure and trusted fashion on (private or public) cloud.

### Trusted SQL & Dataframe


### Trusted Machine Learning
With the trusted Big Data analytics and Machine Learning(ML)/Deep Learning(DL) support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, Spark MLlib, etc.) and distributed deep learning (using BigDL) in a secure and trusted fashion.

See the [PPML ML](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html#trusted-big-data-analytics-and-ml) user guide for more details.

### Trusted Deep Learning
With the Trusted Realtime Compute and ML/DL support, users can run standard Flink stream processing and distributed DL model inference (using Cluster Serving in a secure and trusted fashion
See the [PPML DL](https://bigdl.readthedocs.io/en/latest/doc/PPML/Overview/ppml.html#trusted-realtime-compute-and-ml) user guide for more details.

### Trusted FL (Federated Learning)


## How to use BigDL PPML

## How to expand 
