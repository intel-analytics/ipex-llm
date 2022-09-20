# Introduction for encrypted image

## 1. Install verdictd 
### 1.1 Install rust-lang and golang first.
```bash
# install golang
wget https://dl.google.com/go/go1.16.13.linux-amd64.tar.gz
sha256sum　go1.16.13.linux-amd64.tar.gz
rm -rf /usr/local/go && tar -C /usr/local -xzf go1.16.13.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin
go version

# install cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```
### 1.2 Install EAA KBC
Need to build rootfs and compile EAA component
```bash
git clone https://github.com/containers/attestation-agent
cd attestation-agent
make KBC=eaa_kbc && make install
```
Copy config to `./usr/local/bin/attestation-agent` and modify the config. Please note, `aa_kbc_params= 'eaa_kbc::[host]:[port]'` should have the same host and ip with verdictd.
### 1.3 Install verdictd.
```bash
bash verdictd-deployment.sh
```

## 2. Create encrypted image

Before build encrypted image, you need to make sure Skopeo and KBC module have been installed:
- [Skopeo](https://github.com/containers/skopeo): the command line utility to perform encryption operations.
- [Verdictd](https://github.com/Verdictd): contains KBC modules used to communicate with various KBS. You may follow the tips described in [above part](#install-verdictd) to install Verdictd.

### 2.1 Pull unencrypted image
```bash
${SKOPEO_HOME}/bin/skopeo copy --insecure-policy　docker://docker.io/intelanalytics/bigdl-tdx:latest oci:bigdl-tdx
```
### 2.2 Generate encrypted image
#### Generate the key provider configuration file
```bash
cat <<- EOF >/etc/containerd/ocicrypt/ocicrypt_keyprovider.conf
{
        "key-providers": {
                "attestation-agent": {
                    "grpc": "127.0.0.1:50001"

                }
        }
}
EOF
```

#### Generate a encryption key
```bash
cat <<- EOF >/opt/verdictd/keys/YOUR-KEY-NAME
$YOUR-KEYS
EOF
```
#### Launch Verdictd
```bash
verdictd --client-api 127.0.0.1:50001
```
#### Encrypt and publish image
```bash
export OCICRYPT_KEYPROVIDER_CONFIG=/etc/containerd/ocicrypt/ocicrypt_keyprovider.conf

bash encrypt-image-publish.sh --input-image oci:bigdl-tdx --output-image oci:bigdl-tdx-encrypted --encrypt-key YOUR-KEY-NAME
```

## 3. Create docker registry
#### Deploy docker registry
```bash
bash registry-deployment.sh --registry-name x.x.x.x 
```
#### Pull/push image to docker registry
```bash
# copy registry certs to machine where you will pull/push the image
scp certs/domain.crt /usr/local/share/ca-certificates/registry-name.crt
sudo update-ca-certificates
```
