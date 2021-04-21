## PPML (Privacy Preserving Machine Learning) 
Artificial intelligence on big data is increasingly important to many real-world applications. Many machine learning and data analytics applications are benefiting from the private data in different domains. Most of these applications leverage the private data to offer certain valuable services to the users. But the private data could be repurposed to infer sensitive information, which would jeopardize the privacy of individuals. Privacy-Preserving Machine Learning (PPML) helps address these risks. Using techniques such as cryptography differential privacy, and hardware technologies, PPML aims to protect the privacy of sensitive user data and of the trained model as it performs ML tasks.

Analytics Zoo helps to build PPML applications on top of Intel® SGX Software Guard Extensions (Intel® SGX) and library OSes such as Graphene and Occlum. Analytics Zoo supports big data analytics, machine learning, and cluster serving as PPML applications on top of Intel® SGX, and Graphene/Occlum.

### [Intel® SGX](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
Intel® SGX runs on Intel’s Trusted Execution Environment (TEE), offering hardware-based memory encryption that isolates specific application code and data in memory. Intel® SGX enables user-level code to allocate private regions of memory, called enclaves, which are designed to be protected from processes running at higher privilege levels. 

### [Graphene](https://github.com/oscarlab/graphene)
Graphene is a lightweight guest OS, designed to run a single application with minimal host requirements. Graphene can run applications in an isolated environment with benefits comparable to running a complete OS in a virtual machine -- including guest customization, ease of porting to different OSes, and process migration. Graphene supports native, unmodified Linux applications on any platform. Currently, Graphene runs on Linux and Intel SGX enclaves on Linux platforms. With Intel SGX support, Graphene can secure a critical application in a hardware-encrypted memory region. Graphene can protect applications from a malicious system stack with minimal porting effort.

### [Occlum](https://github.com/occlum/occlum)
Occlum is a memory-safe, multi-process library OS (LibOS) for Intel SGX. As a LibOS, it enables legacy applications to run on SGX with little or even no modifications of source code, thus protecting the confidentiality and integrity of user workloads transparently.

### Analytics-Zoo PPML(Privacy-Preserving Machine Learning) Platform for E2E Big Data and AI using SGX
1. SGX-based Trusted Big Data ML
2. SGX-based Trusted Realtime ML

#### SGX-based Trusted Big Data ML
SGX-based Trusted Big Data ML allows users to run end-to-end big data analytics application and Intel Analytics Zoo and BigDL model training with spark local and distributed cluster on Graphene-SGX.

#### SGX-based Trusted Real Time ML
SGX-based Trusted Realtime ML allows users to run end-to-end Intel Analytics Zoo cluster serving pipeline on both Graphene-SGX and Occlum.


