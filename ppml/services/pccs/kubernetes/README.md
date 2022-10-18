# Deploy BigDL-PCCS on Kubernetes with Helm Charts

## Prerequests

- Please make sure you have a workable **Kubernetes cluster/machine**.
- Please make sure you have a usable https proxy.
- Please make sure your **CPU** is able to run PCCS service, which generate and verify quotes.
- Please make sure you have already installed **[helm](https://helm.sh/)**.
- Please make sure you have an usable PCCS ApiKey for your platform. The PCCS uses this API key to request collaterals from Intel's Provisioning Certificate Service. User needs to subscribe first to obtain an API key. For how to subscribe to Intel Provisioning Certificate Service and receive an API key, goto https://api.portal.trustedservices.intel.com/provisioning-certification and click on 'Subscribe'.

## 1. Start BigDL-PCCS on Kubernetes 
Please make sure current workdir is `kubernetes`.

Then modify parameters in `values.yaml` as following:
```shell
# reset of other parameters in values.yaml is optional, please check according to your environment
pccsIP: your_pccs_ip_to_use_as                    --->   <an_used_ip_address_in_your_subnetwork_to_assign_to_pccs>

# Replace the below parameters according to your environment
apiKey: your_intel_pcs_server_subscription_key_obtained_through_web_registeration
httpsProxyUrl: your_usable_https_proxy_url
countryName: your_country_name
cityName: your_city_name
organizaitonName: your_organizaition_name
commonName: server_fqdn_or_your_name
emailAddress: your_email_address
serverPassword: your_server_password_to_use 
```
Then, deploy BigDL-PCCS on kubernetes:

```bash
kubectl create namespace bigdl-pccs-ehsm-kms
helm install pccs . # pccs can be modified to any name as you like
```
Check the service whether it has successfully been running (it may take seconds):

```bash
kubectl get all -n bigdl-pccs-ehsm-kms

# you will get similar to below
NAME            READY   STATUS        RESTARTS   AGE
pod/pccs-0      1/1     Running       0          18s

NAME           TYPE        CLUSTER-IP      EXTERNAL-IP     PORT(S)     AGE
service/pccs   ClusterIP   10.97.134.251   172.168.0.226   18081/TCP   18s

NAME                    READY   AGE
statefulset.apps/pccs   1/1     18s

```

## 2. Check if pccs service is running and available:
Execute command to check if pccs service is available.
```bash
curl -v -k -G "https://<your_pccs_ip>:<your_pccs_port>/sgx/certification/v3/rootcacrl"

# you will get similar to below if success
...
* Connected to 172.168.0.226 (172.168.0.226) port 18081 (#0)
* ALPN, offering h2
* ALPN, offering http/1.1
* successfully set certificate verify locations:
*   CAfile: /etc/ssl/certs/ca-certificates.crt
  CApath: /etc/ssl/certs
...
```



