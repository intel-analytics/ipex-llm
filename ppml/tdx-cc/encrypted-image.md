## BigDL encrypted image

To build BigDL encrypted image, you need to install:
- [Skopeo](https://github.com/containers/skopeo): the command line utility to perform encryption operations.
- [Attestation Agent](https://github.com/confidential-containers/attestation-agent): contains KBC modules used to communicate with various KBS.

### Pull unencrypted BigDL K8S image
```bash
./skopeo-1.5.2/bin/skopeo copy --insecure-policy　docker://docker.io/intelanalytics/bigdl-k8s:latest oci:bigdl-k8s
```

### Encrypt image with KBS
```bash
OCICRYPT_KEYPROVIDER_CONFIG=ocicrypt.conf ./skopeo-1.5.2/bin/skopeo copy --insecure-policy --encryption-key provider:attestation-agent:test oci:bigdl-k8s oci:bigdl-k8s-encrypted
```
Push image to docker registry
```bash
OCICRYPT_KEYPROVIDER_CONFIG=ocicrypt.conf ./skopeo-1.5.2/bin/skopeo copy --insecure-policy --encryption-key provider:attestation-agent:test oci:bigdl-k8s docker://xxx/bigdl-k8s-encrypted:latest
