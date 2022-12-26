# Secure Your Services

This document is a gentle reminder for enabling security & privacy features for your services. To avoid privacy & security issues during deployment, we recommend Developer/Admin go through this document, which suits users/customers who want to apply BigDL into their production environment (not just for PPML).

## Security in the data lifecycle

Almost all Big Data & AI applications are built upon large-scale datasets, we can simply go through security key steps in the data lifecycle. That is data protection:
* In transit, i.e., network.
* At rest, i.e., storage.
* In use, i.e., computation.

### Secure Network (in transit)

Big Data & AI applications are mainly distributed applications, which means we need to use lots of nodes to run our applications and get jobs done. During that period, not just control flows (commands used to control applications running on different nodes), data partitions (a division of data) may also go through different nodes. So, we need to ensure all network traffic is fully protected.

Talking about secure data transit, TLS is commonly used. The server would provide a private key and certificate chain. To make sure it is fully secured, a complete certificate chain is needed (with two or more certificates built). In addition, SSL/TLS protocol and secure cipher tools would be used. It is also recommended to use forward secrecy and strong key exchange. However, it is general that secure approaches would bring some performance problems. To mitigate these problems, a series of approaches are available, including session resumption, cache, etc. For the details of this section, please see [SSL-and-TLS-Deployment-Best-Practices](https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices).
### Secure Storage (in storage)

Besides network traffic, we also need to ensure data is safely stored in storage. In Big Data & AI applications, data is mainly stored in distributed storage or cloud storage, e.g., HDFS, Ceph and AWS S3 etc. This makes storage security a bit different. We need to ensure each storage node is secured by the correct settings, meanwhile, we need to ensure the whole storage system is secured (network, access control, authentication etc).

### Secure Computation (in use)

Even if data is fully encrypted in transit and storage, we still need to decrypt it when we make some computations. If this stage is not safe, then security & secrets never exist. That's why TEE (SGX/TDX) is so important. In Big Data & AI, applications and data are distributed into different nodes. If any of these nodes are controlled by an adversary, he can simply dump sensitive data from memory or crash your applications. There are lots of security technologies to ensure computation safety. Please check if they are correctly enabled.

## Example: Spark on Kubernetes with data stored on HDFS

WARNING: This example lists minimum security features that should be enabled for your applications. In production, please confirm with your cluster admin or security reviewer.

### Prepare & Manage Your keys

Ensure you are generating, using & managing your keys in the right way. Check with your admin or security reviewer about that. Using Key Management Service (KMS) in your deployment environment is recommended. It will reduce a lot of effort and potential issues.

Back to our example, please prepare SSL & TLS keys based on [SSL & TLS Private Key and Certificate](https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices#1-private-key-and-certificate). Ensure these keys are correctly configured and stored.

In most cases, AES encryption key is not necessary, because Hadoop KMS and Spark will automatically generate keys for your applications or files. However, if you want to use your own keys, please please refer to [generate keys for encryption and decryption](https://docs.microsoft.com/en-us/dotnet/standard/security/generating-keys-for-encryption-and-decryption).

### [HDFS Security](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html)

Please ensure authentication and [access control](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/HdfsPermissionsGuide.html) are correctly configured. Note that HDFS authentication relies on [Kerberos](http://web.mit.edu/kerberos/krb5-1.12/doc/user/user_commands/kinit.html).

Enable [Data_confidentiality](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/SecureMode.html#Data_confidentiality) for network. This will protect PRC, block transfer and http.

When storing sensitive data in HDFS, please enable [Transparent Encryption](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-hdfs/TransparentEncryption.html) in HDFS. This feature ensures all data blocks are encrypted on data nodes.

### [Spark Security](https://spark.apache.org/docs/latest/security.html)

Please ensure [network crypto](https://spark.apache.org/docs/latest/security.html#encryption) and [spark.authenticate](https://spark.apache.org/docs/latest/security.html#spark-rpc-communication-protocol-between-spark-processes) are enabled.

Enable [Local Storage Encryption](https://spark.apache.org/docs/latest/security.html#local-storage-encryption) to protect local temp data.

Enable [SSL](https://spark.apache.org/docs/latest/security.html#ssl-configuration) to secure Spark Webui.

You can enable [Kerberos related settings](https://spark.apache.org/docs/latest/security.html#kerberos) if you have Kerberos service.

### [Kubernetes Security](https://kubernetes.io/docs/concepts/security/)

As a huge resource management service, Kubernetes has lots of security features.

Enable [RBAC](https://kubernetes.io/docs/concepts/security/rbac-good-practices/) to ensure that cluster users and workloads have only access to resources required to execute their roles.

Enable [Encrypting Secret Data at Rest](https://kubernetes.io/docs/tasks/administer-cluster/encrypt-data/) to protect data in rest API.
When mounting key & sensitive configurations into pods, use [Kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/).
