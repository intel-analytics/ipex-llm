# PPML Installation

---

#### OS requirement


```eval_rst
.. note::
    **Hardware requirements**:

     Intel SGX: PPML's features (except Homomorphic Encryption) are mainly built upon Intel SGX. Intel SGX requires Intel CPU with SGX feature, e.g., IceLake (3rd Xeon Platform). `Check if your CPU has SGX feature <https://www.intel.com/content/www/us/en/support/articles/000028173/processors.html>`_
```
```eval_rst
.. note::
    **Supported OS**:

     PPML is thoroughly tested on Ubuntu (18.04/20.04), and should works fine on CentOS/Redhat 8. Note that UEFI (Unified Extensible Firmware Interface) is required for remote attestation registration stage.
```

#### Enable SGX for your Cluster

```eval_rst
.. mermaid::
   
   graph TD
      usesgx{Use SGX?} -- Yes --> installsgx(Install SGX Driver for Node)
      usesgx{Use SGX?} -- No --> he(Homomorphic Encryption)
      installsgx --> installaesm(Install AESM for Node)
      installaesm --> needatt{Need Attestation?}
      needatt -- Yes --> installPCCS(Install PCCS for Cluster)
```


##### Install SGX Driver

Please refer to [Install SGX (Software Guard Extensions) Driver for Xeon Server](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/install_sgx_driver.html).

##### Install AESM (Architectural Enclave Service Manager)

```bash
echo 'deb [arch=amd64] https://download.01.org/intel-sgx/sgx_repo/ubuntu focal main' | sudo tee /etc/apt/sources.list.d/intel-sgx.list > /dev/null
wget -O - https://download.01.org/intel-sgx/sgx_repo/ubuntu/intel-sgx-deb.key | sudo apt-key add -
sudo apt update
sudo apt-get install libsgx-urts libsgx-dcap-ql libsgx-dcap-default-qpl
```

##### Install PCCS (Provisioning Certificate Caching Service) (for attestation)

Please refer to [Intel® Software Guard Extensions Data Center Attestation Primitives (Intel® SGX DCAP): A Quick Install Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/intel-software-guard-extensions-data-center-attestation-primitives-quick-install-guide.html)

Note that PCCS requires Internet connection for downloading certificates from Intel PCS. PCCS is fully [open sourced on Github](https://github.com/intel/SGXDataCenterAttestationPrimitives/blob/master/QuoteGeneration/pccs), you can build your own PCCS based on these codes.

```eval_rst
.. mermaid::
   
   graph BT
      pcs(Intel PCS) --> PCCS
      PCCS --> pcs
      subgraph Internet
         pcs
      end
      subgraph Data Center
         PCCS --> sgx(SGX Server)
         sgx --> PCCS
      end
```

##### Install Kubernetes SGX Plugin (K8S only)

Please refer to [Deploy the Intel SGX Device Plugin for Kubernetes](https://bigdl.readthedocs.io/en/latest/doc/PPML/QuickStart/deploy_intel_sgx_device_plugin_for_kubernetes.html).

### FAQs

1. Is SGX supported on CentOS 6/7?
No. Please upgrade your OS if possible.

2. Do we need Internet connection for SGX node?
No. We can use PCCS for registration and certificate download. Only PCCS need Internet connection.

3. Does PCCS require SGX or other hardware?
No. PCCS can be installed on any server with Internet connection.

4. Can we turn off the attestation?
Of course. But, turning off attestation will break the integrity provided by SGX. Attestation is turned off to simplify installation for quick start.
