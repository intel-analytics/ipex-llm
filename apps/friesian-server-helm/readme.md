# Friesian server deploying with Helm Chart

This example demonstrates how to deploy a Friesian recommender server with Helm Chart on a
Kubernetes cluster.

## Prepare environments

* AVX512 instruction support on nodes that deploy the recall server
* [Preparation steps](./preparation) or a PV contains resource files
* [Helm](https://helm.sh) 3.2.0+
* Kubernetes 1.19+

## Deploy Friesian server

You can get the full installation document from [here](./friesian-helm/readme.md).

## TL;DR

```bash
git clone https://github.com/intel-analytics/BigDL.git
cd BigDL/apps/friesian-server-helm

# create a namespace
kubectl create ns friesian

# create a PVC to store resources and a pod to copy resources
kubectl apply -f './preparation/create-pvc.yaml' -n friesian
kubectl apply -f './preparation/create-volume-pod.yaml' -n friesian

# copy resources to PVC
kubectl cp /path/to/your/resources friesian/volume-pod:/resources

# delete pod no longer used
kubectl delete -f '2. create-volume-pod.yaml' -n friesian

# install friesian server in namespace 'friesian' with name 'my-release'
helm upgrade --install --debug -n friesian my-release ./friesian-helm

```

After installation, you can follow the printed information to check whether the servers are running
properly.