# Trusted Big Data Analytics and ML

Artificial intelligence on big data is increasingly important to many real-world applications. Many machine learning and data analytics applications are benefiting from the private data in different domains. Most of these applications leverage the private data to offer certain valuable services to the users. But the private data could be repurposed to infer sensitive information, which would jeopardize the privacy of individuals. Privacy-Preserving Machine Learning (PPML) helps address these risks. Using techniques such as cryptography differential privacy, and hardware technologies, PPML aims to protect the privacy of sensitive user data and of the trained model as it performs ML tasks.

BigDL helps to build PPML applications (including big data analytics, machine learning, and cluster serving etc) on top of Intel® SGX Software Guard Extensions (Intel® SGX) and library OSes such as Graphene and Occlum. In the current release, two types of trusted Big Data AI applications are supported:

1. Big Data analytics and ML/DL (supporting [Apache Spark](https://spark.apache.org/) and [BigDL](https://github.com/intel-analytics/BigDL))
2. Realtime compute and ML/DL (supporting [Apache Flink](https://flink.apache.org/) and BigDL [Cluster Serving](https://www.usenix.org/conference/opml20/presentation/song))

## [1. Trusted Big Data ML](https://github.com/intel-analytics/BigDL/tree/branch-2.0/ppml/trusted-big-data-ml)

With the trusted Big Data analytics and ML/DL support, users can run standard Spark data analysis (such as Spark SQL, Dataframe, MLlib, etc.) and distributed deep learning (using BigDL) in a secure and trusted fashion.

## [2. Trusted Real Time ML](https://github.com/intel-analytics/BigDL/tree/branch-2.0/ppml/trusted-realtime-ml/scala)

With the trusted realtime compute and ML/DL support, users can run standard Flink stream processing and distributed DL model inference (using Cluster Serving) in a secure and trusted fashion.

## 3. Intel SGX and LibOS

### [Intel® SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)

Intel® SGX runs on Intel’s Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. Intel® SGX enables user-level code to allocate private regions of memory, called enclaves, which are designed to be protected from processes running at higher privilege levels.

### [Graphene-SGX](https://github.com/oscarlab/graphene)

Graphene is a lightweight guest OS, designed to run a single application with minimal host requirements. Graphene can run applications in an isolated environment with benefits comparable to running a complete OS in a virtual machine -- including guest customization, ease of porting to different OSes, and process migration. Graphene supports native, unmodified Linux applications on any platform. Currently, Graphene runs on Linux and Intel SGX enclaves on Linux platforms. With Intel SGX support, Graphene can secure a critical application in a hardware-encrypted memory region. Graphene can protect applications from a malicious system stack with minimal porting effort.

### [Occlum](https://github.com/occlum/occlum)

Occlum is a memory-safe, multi-process library OS (LibOS) for Intel SGX. As a LibOS, it enables legacy applications to run on SGX with little or even no modifications of source code, thus protecting the confidentiality and integrity of user workloads transparently.
