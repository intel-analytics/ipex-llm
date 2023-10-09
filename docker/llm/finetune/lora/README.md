## Run BF16-Optimized Lora Finetuning on Kubernetes with OneCCL

[Alpaca Lora](https://github.com/tloen/alpaca-lora/tree/main) uses [low-rank adaption](https://arxiv.org/pdf/2106.09685.pdf) to speed up the finetuning process of base model [Llama2-7b](https://huggingface.co/meta-llama/Llama-2-7b), and tries to reproduce the standard Alpaca, a general finetuned LLM. This is on top of Hugging Face transformers with Pytorch backend, which natively requires a number of expensive GPU resources and takes significant time.

By constract, BigDL here provides a CPU optimization to accelerate the lora finetuning of Llama2-7b, in the power of mixed-precision and distributed training. Detailedly, [Intel OneCCL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/oneccl.html), an available Hugging Face backend, is able to speed up the Pytorch computation with BF16 datatype on CPUs, as well as parallel processing on Kubernetes enabled by [Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html). 

The architecture is illustrated in the following:

![image](https://github.com/Jasonzzt/BigDL/assets/60865256/b66416bc-ad07-49af-8cb0-8967dffb5f58)

As above, BigDL implements its MPI training build on [Kubeflow MPI operator](https://github.com/kubeflow/mpi-operator/tree/master), which encapsulates the deployment as MPIJob CRD, and assists users to handle the construction of a MPI worker cluster on Kubernetes, such as public key distribution, SSH connection, and log collection. 

Now, let's go to deploy a Lora finetuning to create a LLM from Llama2-7b.

**Note: Please make sure you have already have an available Kubernetes infrastructure and NFS shared storage, and install [Helm CLI](https://helm.sh/docs/helm/helm_install/) for Kubernetes job submission.**

### 1. Install Kubeflow MPI Operator

Follow [here](https://github.com/kubeflow/mpi-operator/tree/master#installation) to install a Kubeflow MPI operator in your Kubernetes, which will listen and receive the following MPIJob request at backend.

### 2. Download Image, Base Model and Finetuning Data

Follow [here](https://github.com/intel-analytics/BigDL/tree/main/docker/llm/finetune/lora/docker#prepare-bigdl-image-for-lora-finetuning) to prepare BigDL Lora Finetuning image in your cluster.

As finetuning is from a base model, first download [Llama2-7b model from the public download site of Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b). Then, download [cleaned alpaca data](https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_cleaned_archive.json), which contains all kinds of general knowledge and has already been cleaned. Next, move the downloaded files to a shared directory on your NFS server.

### 3. Deploy through Helm Chart

You are allowed to edit and experiment with different parameters in `./kubernetes/values.yaml` to improve finetuning performance and accuracy. For example, you can adjust `trainerNum` and `cpuPerPod` according to node and CPU core numbers in your cluster to make full use of these resources, and different `microBatchSize` result in different training speed and loss (here note that `microBatchSize`×`trainerNum` should not more than 128, as it is the batch size).

**Note: `dataSubPath` and `modelSubPath` need to have the same names as files under the NFS directory in step 2.**

After preparing parameters in `./kubernetes/values.yaml`, submit the job as beflow:

```bash
cd ./kubernetes
helm install bigdl-lora-finetuning .
```

### 4. Check Deployment
```bash
kubectl get all -n bigdl-lora-finetuning # you will see launcher and worker pods running
```

### 5. Check Finetuning Process

After deploying successfully, you can find a launcher pod, and then go inside this pod and check the logs collected from all workers.

```bash
kubectl get all -n bigdl-lora-finetuning # you will see a launcher pod
kubectl exec -it <launcher_pod_name> bash -n bigdl-ppml-finetuning # enter launcher pod
cat launcher.log # display logs collected from other workers
```

From the log, you can see whether finetuning process has been invoked successfully in all MPI worker pods, and a progress bar with finetuning speed and estimated time will be showed after some data preprocessing steps (this may take quiet a while).

For the fine-tuned model, it is written by the worker 0 (who holds rank 0), so you can find the model output inside the pod, which can be saved to host by command tools like `kubectl cp` or `scp`.


## To run in TDX-CoCo and enable Remote Attestation API

You can deploy this workload in TDX CoCo and enable Remote Attestation API Serving with setting `TEEMode` in `./kubernetes/values.yaml` to `tdx`. The main diffences are it's need to execute the pods as root and mount TDX device, and a flask service is responsible for generating launcher's quote and collecting workers' quotes. 

### (Optional) Enable TLS
To enable TLS in Remote Attestation API Serving, you should provide a TLS certificate and setting `enableTLS` ( to `true` ), `base64ServerCrt` and `base64ServerKey` in `./kubernetes/values.yaml`.
```bash
# Generate a self-signed TLS certificate (DEBUG USE ONLY)
export COUNTRY_NAME=your_country_name
export CITY_NAME=your_city_name
export ORGANIZATION_NAME=your_organization_name
export COMMON_NAME=your_common_name
export EMAIL_ADDRESS=your_email_address

openssl req -x509 -newkey rsa:4096 -nodes -out server.crt -keyout server.key -days 365 -subj "/C=$COUNTRY_NAME/ST=$CITY_NAME/L=$CITY_NAME/O=$ORGANIZATION_NAME/OU=$ORGANIZATION_NAME/CN=$COMMON_NAME/emailAddress=$EMAIL_ADDRESS/"

# Calculate Base64 format string in values.yaml
cat server.crt | base64 -w 0 # Set in base64ServerCrt
cat server.key | base64 -w 0 # Set in base64ServerKey
```

To use RA Rest API, you need to get the IP of job-launcher:
``` bash
kubectl get all -n bigdl-lora-finetuning 
```
You will find a line like:
```bash
service/bigdl-lora-finetuning-launcher-attestation-api-service   ClusterIP   10.109.87.248   <none>        9870/TCP   17m
```
Here are IP and port of the Remote Attestation API service.

The RA Rest API are listed below:
### 1. Generate launcher's quote
```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_report_data": "<your_user_report_data>"}' http://<your_ra_api_service_ip>:<your_ra_api_service_port>/gen_quote
```

Example responce:

```json
{"quote":"BAACAIEAAAAAAAA..."}
```
### 2. Collect all cluster components' quotes (launcher and workers)
```bash
curl -X POST -H "Content-Type: application/json" -d '{"user_report_data": "<your_user_report_data>"}' http://<your_ra_api_service_ip>:<your_ra_api_service_port>/attest
```

Example responce:

```json
{"quote_list":{"bigdl-lora-finetuning-job-worker-0":"BAACAIEAAAAAAA...","bigdl-lora-finetuning-job-worker-1":"BAACAIEAAAAAAA...","launcher":"BAACAIEAAAAAA..."}}
```

