This is a temporary README for internal test, which is under refining.

## Install Kubeflow MPI Operator
```bash
kubectl apply -f https://raw.githubusercontent.com/kubeflow/mpi-operator/master/deploy/v2beta1/mpi-operator.yaml
```

## Download Data and Huggingface Model
```bash
wget https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json
```

Public download site of hf llama 7b model is: https://huggingface.co/decapoda-research/llama-7b-hf/tree/main (internally, we have ftp etc. archives to speed up downloading)

Note: please make sure the subPath in `./k8s/values.yaml` have same values as the download files' names/

## Deploy through Helm Chart
```bash
cd ./k8s
helm install lora-on-kubeflow .
```

## Check Deployment
```bash
kubectl get all -n bigdl-ppml-finetuning # you will see launcher and worker pods running
```

## Check Training
```bash
kubectl exec -it <launcher_pod_name> bash -n bigdl-ppml-finetuning # enter launcher pod
cat launcher.log # wait for progress bar like: 1/582 xxx s/it
```
