# Secure Your Services

This document is a gentle remainder for enabling security & privacy features for your services. To avoid privacy & security issues during deployment, we recommend Developer/Admin to go through this document. It apples for user/customer who want to apply BigDL into their production environment (not just for PPML).

## Security in data lifecycle

Almost all Big Data & AI applications are built upon large scale dataset, we can simply go through security key steps in data lifecycle. That is data protection in transit, in storage, and in use.

### Secure Network (in transit)

Big Data & AI applications are mainly distributed applications. That means we need to use lots of nodes to run our applications and get jobs done. During that period, not just control flows (command used to control applications running on different nodes), data partitions (a division of data) may also go through different nodes. So, we need to ensure all network traffic are fully protected.

### Secure Storage (in storage)

Beside network traffic, we also need to ensure data is safely stored in hard disk. In Big Data & AI applications, data is mainly stored in distributed storage or cloud storage, e.g., HDFS, Ceph and AWS S3 etc. That makes storage security a bit different node. We need to ensure each storage node is secured by correct settings, meanwhile we need to ensure the whole storage system is secured (network, access control, authentication etc).

### Secure Computation (in use)

Even if data are fully encrypted in transit and storage, we still need to decrypt it when we make some computation. If this stage is not safe, then security & secret never exists. That's why TEE (SGX/TDX) is so important. In Big Data & AI, applications and data are distributed into different nodes. If any of these nodes are controlled by an adversary, he can simply dump sensitive data from memory or crash your applications. There are lots of security technologies to ensure computation safety. Please check if they are correctly enabled.

## Example: Spark on Kubernetes with data stored on HDFS

WARNING: This example lists minimum security features that should be enabled for your applications. In production, please confirm with your cluster admin or security reviewer.

### [HDFS Security](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html)

Please ensure authentication and [access control](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsPermissionsGuide.html) is correctly configured. Note that HDFS authentication relay on [Kerberos](http://web.mit.edu/kerberos/krb5-1.12/doc/user/user_commands/kinit.html).

Enable [Data_confidentiality](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html#Data_confidentiality) for network. This will protect PRC, block transfer and http.

When storing sensitive data in HDFS, please enable [Transparent Encryption](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/TransparentEncryption.html) in HDFS. This feature ensured all data blocks are encrypted on data nodes.

### [Spark Security](https://spark.apache.org/docs/latest/security.html)

Please ensure [network crypto](https://spark.apache.org/docs/latest/security.html#encryption) and [spark.authenticate](https://spark.apache.org/docs/latest/security.html#spark-rpc-communication-protocol-between-spark-processes) are enabled.

Enable [Local Storage Encryption](https://spark.apache.org/docs/latest/security.html#local-storage-encryption) to protect local temp data.

Enable [SSL](https://spark.apache.org/docs/latest/security.html#ssl-configuration) to secure Spark Webui.

You can enable [Kerberos related settings](https://spark.apache.org/docs/latest/security.html#kerberos) if your have Kerberos service.

### [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

As a huge resource management service, Kubernetes has lots of security features.

Enable [RBAC](https://kubernetes.io/docs/concepts/security/rbac-good-practices/) to ensure that cluster users and workloads have only the access to resources required to execute their roles.

Enable [Encrypting Secret Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/) to protect data in rest API.

When mount key & sensitive configurations into pod, use [Kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/).
