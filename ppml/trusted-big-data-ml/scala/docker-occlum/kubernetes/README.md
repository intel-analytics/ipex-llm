# Spark 3.0.0 on K8S with Occlum

## Pre-prerequisites

Check Kubernetes env or Install Kubernetes from [wiki](https://kubernetes.io/zh/docs/setup/production-environment)

1. Initialize and enable taint for master node

```bash
swapoff -a && free -m
kubeadm init --v=5 --node-name=master-node --pod-network-cidr=10.244.0.0/16

mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

kubectl taint nodes --all node-role.kubernetes.io/master-
```

2. Deploy flannel and ingress-nginx service

```bash
kubectl apply -f flannel/deploy.yaml
kubectl apply -f ingress-nginx/deploy.yaml
```

3. Add spark account

```bash
kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default
```

## Run Spark executor in Occlum:

1. Download [Spark 3.0.0](https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop2.7.tgz), and setup `SPARK_HOME`.
2. Modify `${kubernetes_master_url}` to your k8s master url in the `run_pi.sh `
3. Modify `executor.yaml`

```
./run_pi.sh
```
