# Deploy BigDL-PCCS with Docker
## Prerequests

- Please make sure you have a workable **Kubernetes cluster/machine**.
- Please make sure PCCS has Internet access. If you are using a proxy, make sure https proxy is correctly set. Otherwise set `HTTPS_PROXY_URL `in `install-bigdl-pccs.sh` to `""`.
- Please make sure you have an usable PCCS ApiKey for your platform. The PCCS uses this API key to request collaterals from Intel's Provisioning Certificate Service. User needs to subscribe first to obtain an API key. For how to subscribe to Intel Provisioning Certificate Service and receive an API key, goto https://api.portal.trustedservices.intel.com/provisioning-certification and click on 'Subscribe'.

## 1. Pull/Build the PCCS Image

We encapsulate host PCCS service into a docker image, which enables a user-friendly container-service.

Download image as below:

```bash
docker pull intelanalytics/pccs:0.3.0-SNAPSHOT
```

Or you are allowed to build the image manually:

```bash
cd ../pccs
# configure build parameters in build-docker-image.sh
bash build-docker-image.sh
cd ../kubernetes
```
## 2. Start BigDL-PCCS on Kubernetes 
### 2.1 Determine PCCS IP address
First of all, Please note that an **IP address that unused in your subnetwork** is needed to be used as PCCS IP. \
**Especially,** this IP address chosen for PCCS **SHOULD NOT** be the real machine IP address. \
You could check if the IP address is available like this.

```bash
# assume your IP address is 1.2.3.4, and you want to use 1.2.3.226 as PCCS IP
ping 1.2.3.226

# information below means 1.2.3.226 is expected to be an appropriate IP addess for PCCS. 
# otherwise, you are supposed to test another one.
PING 1.2.3.226 (1.2.3.226) 56(84) bytes of data.
From 1.2.3.4 icmp_seq=1 Destination Host Unreachable
From 1.2.3.4 icmp_seq=2 Destination Host Unreachable
From 1.2.3.4 icmp_seq=3 Destination Host Unreachable
........
```

### 2.2 Modify the script and deploy BigDL-PCCS
Then, modify parameters in `install-bigdl-pccs.sh` as following, and `pccsIP` should be the IP address you have determined in step 2.1.
```bash
# reset of other parameters in bigdl-pccs.yaml is optional, please check according to your environment
pccsIP: your_pccs_ip_to_use_as                    
user_password: a_password_for_pccs_user
admin_password: a_password_for_pccs_admin
# replace the below parameters according to your environment
apiKey: your_intel_pcs_server_subscription_key_obtained_through_web_registeration
countryName: your_country_name
cityName: your_city_name
organizaitonName: your_organizaition_name
commonName: server_fqdn_or_your_name
emailAddress: your_email_address
httpsCertPassword: your_https_cert_password_to_use 
```
Then, deploy BigDL-PCCS on kubernetes:

```bash
bash install-bigdl-pccs.sh
```
Check the service whether it has successfully been running (it may take seconds):

```bash
kubectl get all -n bigdl-pccs

# you will get similar to below
NAME            READY   STATUS        RESTARTS   AGE
pod/pccs-0      1/1     Running       0          18s

NAME           TYPE        CLUSTER-IP      EXTERNAL-IP     PORT(S)     AGE
service/pccs   ClusterIP   1.7.4.251   1.2.3.4   18081/TCP   18s

NAME                    READY   AGE
statefulset.apps/pccs   1/1     18s

```

## 3. Check if pccs service is running and available:
Execute command to check if pccs service is available.
```bash
curl -v -k -G "https://<your_pccs_ip>:<your_pccs_port>/sgx/certification/v3/rootcacrl"

# you will get similar to below if success

* Uses proxy env variable no_proxy == '10.239.45.10:8081,10.112.231.51,10.239.45.10,172.168.0.205'
*   Trying 1.2.3.4:18081...
* TCP_NODELAY set
* Connected to 1.2.3.4 (1.2.3.4) port 18081 (#0)
* ALPN, offering h2
* ALPN, offering http/1.1
* successfully set certificate verify locations:
*   CAfile: /etc/ssl/certs/ca-certificates.crt
  CApath: /etc/ssl/certs
* TLSv1.3 (OUT), TLS handshake, Client hello (1):
* TLSv1.3 (IN), TLS handshake, Server hello (2):
* TLSv1.3 (IN), TLS handshake, Encrypted Extensions (8):
* TLSv1.3 (IN), TLS handshake, Certificate (11):
* TLSv1.3 (IN), TLS handshake, CERT verify (15):
* TLSv1.3 (IN), TLS handshake, Finished (20):
* TLSv1.3 (OUT), TLS change cipher, Change cipher spec (1):
* TLSv1.3 (OUT), TLS handshake, Finished (20):
* SSL connection using TLSv1.3 / TLS_AES_256_GCM_SHA384
* ALPN, server accepted to use http/1.1
* Server certificate:
*  subject: C=cn; ST=nanjing; L=nanjing; O=intel; OU=intel; CN=liyao; emailAddress=yao3.li@intel.com
*  start date: Oct 17 08:14:42 2022 GMT
*  expire date: Oct 17 08:14:42 2023 GMT
*  issuer: C=cn; ST=nanjing; L=nanjing; O=intel; OU=intel; CN=liyao; emailAddress=yao3.li@intel.com
*  SSL certificate verify result: self signed certificate (18), continuing anyway.
> GET /sgx/certification/v3/rootcacrl HTTP/1.1
> Host: 1.2.3.4:18081
> User-Agent: curl/7.68.0
> Accept: */*
>
* TLSv1.3 (IN), TLS handshake, Newsession Ticket (4):
* TLSv1.3 (IN), TLS handshake, Newsession Ticket (4):
* old SSL session ID is stale, removing
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< X-Powered-By: Express
< Request-ID: 64371451f83842079bded0b228fb7d1a
< Content-Type: application/x-pem-file; charset=utf-8
< Content-Length: 586
< ETag: W/"24a-lXdmj38gN2RweL6On8KEs2rk9To"
< Date: Tue, 18 Oct 2022 01:46:43 GMT
< Connection: keep-alive
< Keep-Alive: timeout=5
<
* Connection #0 to host 1.2.3.4 left intact
**308201213081c80......**
```

The service is ready once you receive the root CA.

## 4. Delete BigDL-PCCS from Kuberbets
Run the uninstall script as below
```bash
bash uninstall-bigdl-pccs.sh

# you will get similar to below if success
service "pccs" deleted
statefulset.apps "pccs" deleted
namespace "bigdl-pccs" deleted

``` 

## 5. Register a machine to PCCS
1. According to your system version, download PCKIDRetrievalTool from [here](https://download.01.org/intel-sgx/sgx-dcap/1.15/linux/distro/)
2. (Optional)If you set proxy in PCCS, then:
```bash
export no_proxy=your_host_ip
```
3. modify PCKIDRetrievalTool_v1.15.100.3/network_setting.conf
```conf
# support V4 version PCCS
PCCS_URL=https://your_pccs_ip:your_pccs_port/sgx/certification/v4/platforms
# To accept insecure HTTPS cert, set this option to FALSE
USE_SECURE_CERT=FALSE
```
4. modify /etc/sgx_default_qcnl.conf
```conf
//PCCS server address
"pccs_url": "https://your_pccs_ip:your_pccs_port/sgx/certification/v4/",

// To accept insecure HTTPS certificate, set this option to false
"use_secure_cert": false,
```
5. use PCKIDRetrievalTool
```bash
./PCKIDRetrievalTool -user_token your_user_password
```