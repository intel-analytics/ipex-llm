# Deploy BigDL Remote Attestation Service on Kubernetes

```bash
# 1. Create namespace bigdl-remote-attestation-service if not exists
kubectl create namespace bigdl-remote-attestation-service

# 2. Configure PCCS_URL, ATTESTATION_SERVICE_HOST and ATTESTATION_SERVICE_PORT in bigdl-attestation-service.yaml
vi bigdl-attestation-service.yaml

# 3. Apply bigdl-attestation-service.yaml
kubectl apply -f bigdl-attestation-service.yaml
```