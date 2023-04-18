## Prerequests

- Please make sure you have Docker installed.
- Please make sure this PCCS container has Internet access. If you are using a proxy, make sure https proxy is correctly set. Otherwise set `HTTPS_PROXY_URL `in `run-docker-container.sh` to `""`.
- Please make sure you have an usable PCCS ApiKey for your platform. The PCCS uses this API key to request collaterals from Intel's Provisioning Certificate Service. User needs to subscribe first to obtain an API key. For how to subscribe to Intel Provisioning Certificate Service and receive an API key, goto https://api.portal.trustedservices.intel.com/provisioning-certification and click on 'Subscribe'.
## 1. Pull/Build container image
Download image as below:

```bash
docker pull intelanalytics/pccs:0.3.0-SNAPSHOT
```

Or you are allowed to build the image manually:
```
# set the arguments inside the build script first
bash build-docker-image.sh
```

## 2. Run container
```
# set the arguments inside the build script first
bash run-docker-container.sh
```

## 3 . Check if pccs service is running and available:
```
docker logs -f <your_pccs_container_name>
```

Output:

```
2021-08-01 20:54:24.700 [info]: DB Migration -- Update pcs_version table
2021-08-01 20:54:24.706 [info]: DB Migration -- update pck_crl.pck_crl from HEX string to BINARY
2021-08-01 20:54:24.709 [info]: DB Migration -- update pcs_certificates.crl from HEX string to BINARY
2021-08-01 20:54:24.711 [info]: DB Migration -- update platforms(platform_manifest,enc_ppid) from HEX string to BINARY
2021-08-01 20:54:24.713 [info]: DB Migration -- update platforms_registered(platform_manifest,enc_ppid) from HEX string to BINARY
2021-08-01 20:54:24.715 [info]: DB Migration -- Done.
2021-08-01 20:54:24.831 [info]: HTTPS Server is running on: https://localhost:<your_pccs_port>

```

Execute command:
```
curl -v -k -G "https://<your_pccs_ip>:<your_pccs_port>/sgx/certification/v4/rootcacrl"
```
to check if pccs service is available.

## 4. Register a machine to PCCS

Please refer to [this](https://github.com/intel-analytics/BigDL/tree/main/ppml/services/pccs/kubernetes#5-register-a-machine-to-pccs)

## 5. Stop container:
```
docker stop <your_pccs_container_name>
```

