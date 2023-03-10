# Encrypt data wih EHSM on TDX
## 1.1 Create container

```bash
kubectl apply -f ehsm_pod.yaml
```

## 1.2 enroll, generate key, encrypt and decrypt
```bash
# Enroll
curl -v -k -G "https://<kms_ip>:9000/ehsm?Action=Enroll"

......

{"code":200,"message":"successful","result":{"apikey":"jNev7QHpvJ55LRG6Ndubeb64NNBX6Pb9","appid":"19f3f554-8dd0-4ae9-a55a-29793eb88231"}}

# Generatekeys
export appid=19f3f554-8dd0-4ae9-a55a-29793eb88231
export apikey=jNev7QHpvJ55LRG6Ndubeb64NNBX6Pb9
kubectl exec -it bigdl-tdx-ehsm -- bash -c "bash /home/entrypoint.sh generatekeys $appid $apikey"
```
