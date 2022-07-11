Set up K8s cluster: placeholder

Set up K8s-SGX plugin: deploy_intel_sgx_device_plugin_for_k8s

Set up Attestation service: placeholder

Set up KMS (key management service): ehsm-kms

(Optional) Set up K8s Monitioring: bigdl-ppml-sgx-k8s-prometheus/README.md

key/secret

Prepare BigDL PPML Docker Image

Pull Docker image from Dockerhub
docker pull intelanalytics/bigdl-ppml-trusted-big-data-ml-scala-graphene:2.1.0-SNAPSHOT
Alternatively, you can build Docker image from Dockerfile (this will take some time):
cd trusted-big-data-ml/python/docker-graphene
./build-docker-image.sh
