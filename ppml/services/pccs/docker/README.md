## 1. Build container image
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
curl -v -k -G "https://<your_pccs_ip>:<your_pccs_port>/sgx/certification/v3/rootcacrl"
```
to check if pccs service is available.

## 4. Stop container:
```
docker stop <your_pccs_container_name>
```

