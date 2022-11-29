# Ensure Integrity and Build Trust with Attestation

The process of validating the integrity of a computing device such as a server needed for trusted computing. It is widely used in a Trusted Execution Environment (TEE) or Trusted Platform Module (TPM) for ensuring integrity and building trust.

### Attestation Basic

The basic idea of attestation is to verify:
1. The platform is secured. Trusted Computing Base (TCB) is secured.
2. Running in TEE/TPM.
3. Application is as expected (same hash or HMAC).

Local or remote attestation:

* Verifying a local enclave (TEE env) on the same node/server is called local attestation.
* Verifying a remote enclave on another node/server is called remote attestation.

Due to platform differences, Intel SGX has 2 kinds of attestations:

1. Elliptic Curve Digital Signature Algorithm (ECDSA) Attestation for 3rd generation Intel® Xeon® Scalable processors and selected Intel® Xeon® E3 processors.
2. Intel® Enhanced Privacy ID (Intel® EPID) Attestation for desktop and Xeon E3 processors, and selected Intel® Xeon® E processor.

*Note that SGX attestation mentioned in BigDL PPML should be ECDSA attestation with DCAP.*

The basic workflow of attestation:

```eval_rst
.. mermaid::
   
   sequenceDiagram
      Verifier->>App in SGX: Challenge(Prove YourSelf)
      Note right of App in SGX: Generate Quote(Signed Context)
      App in SGX->>Verifier: Evidence(App Quote)
      Note left of Verifier: Verify Quote
      Verifier ->>App in SGX: Response(Pass/Fail)
```

The key steps in attestation:
* Quote Generation. Generate a Quote/Evidence with SDK/API. This quote is signed by a pre-defined key, and it cannot be modified. You can add 128bits user data into a SGX quote.
* Quote Verification. Verify a Quote/Evidence with SDK/API. 

### Attestation in E2E PPML applications

Attestation is not hard if you are running a new written application. Because you can directly integrate `quote generation` and `quote verification` into your application code. However, if you are migrating an existing application, attestation may cause some additional effort. Especially, when you are running distributed applications like PPML applications in multi-nodes. That means you have to add attestation into your distributed applications or frameworks, e.g., add attestation when modules running on different nodes build connections.

To avoid such changes, we can utilize a third-party attestation service to offload `quote verification` from your existing applications. This service will help us to verify if a running application is as expected.

#### Attestation Service

When working with an attestation service, we can define a policy/requirement for each application. During application initialization (server or worker), we can require each module to generate its quote and send it to an attestation service. This attestation service will check these quotes based on pre-defined policy/requirement, then send back responses (`success/fail`). If we get a `success` result, we keep starting this module. Otherwise, we just quit or kill this module.

```eval_rst
.. mermaid::
   
   graph TD
      Admin --Policy--> as(Attestation Service)
      subgraph Production Env/Cloud
         sgxserver(Server in SGX) -.- sgxworker1
         sgxserver(Server in SGX) -.- sgxworker2
         sgxworker1(Worker1 in SGX)
         sgxworker2(Worker2 in SGX)
      end
      sgxserver --Quote--> as
      sgxworker1 --Quote--> as
      sgxworker2 --Quote--> as
      as --response-->sgxserver
      as --response-->sgxworker1
      as --response-->sgxworker2
```

With this attestation service design, we can avoid adding malicious applications or modules to distributed applications.

#### Attestation Service from Cloud Service Provider (CSP)

Azure provides an Attestation Service for applications running in TEE VM or containers provided by Azure. Before we submit our applications to a cloud service, we need to verify the identity and security posture of the platform. Azure Attestation receives evidence from the platform, validates it with security standards, evaluates it against configurable policies, and produces an attestation token for claims-based applications.

The involved actors in Azure Attestation workflow:
* Relying party: The component which relies on Azure Attestation to verify enclave validity.
* Client: The component which collects information from an enclave and sends requests to Azure Attestation.
* Azure Attestation: The component which accepts enclave evidence from client, validates it and returns attestation token to the client

![Azure Attestation Workflow](https://learn.microsoft.com/en-us/azure/attestation/media/sgx-validation-flow.png)

Here are the general steps in a typical SGX enclave attestation workflow (using Azure Attestation): The client collect the evidence from the enclave by generating a quote and send it to an URI which refers to an instance of Azure Attestation. Azure Attestation validates the submitted information and evaluates it against a configured policy. If the verification succeeds, Azure Attestation issues an attestation token and returns it to the client. The client sends the attestation token to relying party. The relying party calls public key metadata endpoint of Azure Attestation to retrieve signing certificates. The relying party then verifies the signature of the attestation token and ensures the enclave trustworthiness. 

### Advanced Usage

During remote attestation, the attestation protocol will build a secure channel. It can help build [TLS connection with integrity](https://arxiv.org/pdf/1801.05863.pdf). Meanwhile, attestation can be [integrated with HTTP protocol to provide trusted end-to-end web service](https://arxiv.org/abs/2205.01052).

### References

1. https://sgx101.gitbook.io/sgx101/sgx-bootstrap/attestation
2. https://www.intel.com/content/www/us/en/developer/articles/technical/quote-verification-attestation-with-intel-sgx-dcap.html
3. https://download.01.org/intel-sgx/sgx-dcap/1.9/linux/docs/Intel_SGX_DCAP_ECDSA_Orientation.pdf
4. https://azure.microsoft.com/en-us/products/azure-attestation/
5. https://en.wikipedia.org/wiki/Trusted_Computing
6. [Integrating Intel SGX Remote Attestation with Transport Layer Security](https://arxiv.org/pdf/1801.05863.pdf)
7. [HTTPA/2: a Trusted End-to-End Protocol for Web Services](https://arxiv.org/abs/2205.01052)
