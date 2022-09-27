# Run DeepRec with BigDL
Here we demonstrate how to integrate [DeepRec](https://github.com/alibaba/DeepRec) into BigDL so as to easily build end-to-end recommendation pipelines for Spark data processing and DeepRec model training.

See [here](https://github.com/alibaba/DeepRec/tree/main/modelzoo/wide_and_deep) for the original Wide & Deep training example in DeepRec. This BigDL example uses BigDL Friesian for distribtued feature engineering and BigDL Orca for launching DeepRec distributed training on the Kubernetes cluster.

## 1. Environment Preparation
1. Enter the client node of the k8s cluster.
2. Pull BigDL image on the client node (probably as well as on all nodes of the k8s cluster):
```bash
docker pull intelanalytics/bigdl-k8s
```
3. Create the k8s client container on the client node:
```bash
sudo docker run -itd --net=host \
    -v /mnt/disk1/nfsdata/default-nfsvolumeclaim-pvc-fe6e673d-8d2f-4653-b805-ca2501df0c2a:/bigdl2.0/data \
    -v /dev/shm:/dev/shm \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v /root/.pip:/root/.pip \
    intelanalytics/bigdl-k8s:latest bash
```
Remarks:
- If you just want to run in local mode, you don't need to mount kubernetes related folders into the container.
- We use Network File System (NFS) for k8s and thus we mount the NFS on the client node into the container. Your are recommended to put your scripts and data into NFS.

4. Enter the client container.
```bash
docker exec -it container_id bash
```
5. Install [Anaconda](https://www.anaconda.com/distribution/#linux) in the container and create a Python 3.7 environment:
```bash
conda create --name bigdl python=3.7
```
6. Install the necessary packages in the conda environment:
```bash
conda activate bigdl

pip install --pre --upgrade bigdl-friesian-spark3[train]
pip install protobuf==3.19.4
pip install numpy==1.18.5
```
7. Download and install DeepRec in the conda environment:
```bash
wget https://deeprec-whl.oss-cn-beijing.aliyuncs.com/tensorflow-1.15.5%2Bdeeprec2204-220614%2Bglibc-cp37-cp37m-linux_x86_64.whl
pip install tensorflow-1.15.5+deeprec2204-220614+glibc-cp37-cp37m-linux_x86_64.whl

# The corresponding stock TensorFlow version should be 1.15.5.
```
8. Run the program `wdl.py`. You may need to change the NFS configurations in `init_orca_context` according to your cluster settings.

## 2. Data Preparation
Please refer to the [README](https://github.com/alibaba/DeepRec/tree/main/modelzoo/wide_and_deep/data) of DeepRec's WDL example to download the dataset. Put `train.csv` and `eval.csv` under the same folder. The files should be accessible to all nodes in the cluster (e.g. in NFS).

## 3. Train DeepRec WDL
- Local mode:
```bash
python wdl.py \
    --instances_per_node 3 \
    --data_location /folder/path/to/train/and/test/files \
    --checkpoint /path/to/save/model/checkpoint \
    --ev True \
    --ev_filter counter \
    --smartstaged False \
    --emb_fusion False \
    --optimizer adam
```
- K8s mode:
```bash
python wdl.py \
    --cluster_mode k8s \
    --master k8s://https://ip:port \
    --num_nodes 3 \
    --data_location /folder/path/to/train/and/test/files \
    --checkpoint /path/to/save/model/checkpoint \
    --ev True \
    --ev_filter counter \
    --smartstaged False \
    --emb_fusion False \
    --optimizer adam
```

For DeepRec related arguments, please refer to the original example for more description.

We put data in NFS and save the model checkpoint to NFS as well. You need to change the data_location and checkpoint paths in the above command.

**Additional Options**:
- `num_nodes`: The number of nodes to use in the cluster. Default to be 1 for local mode.
- `cores`: The number of cpu cores to use on each node. Default to be 8.
- `instances_per_node`: The number of ps and worker instances to run on each node. Default to be 1. For local mode, this value needs to be no less than 2 as there must be at least 1 worker and 1 parameter server.
- `num_ps`: The number of parameter servers to use. Default to be 1.
- `in_memory`: Whether to run the example based on in-memory data ingestion. Default to be False. Add `--in_memory` in the running command to enable in-memory data transfer.

## 4. Evaluation Results
Evaluation results on the test dataset of each worker will be printed at the end:
```
(RayWorker pid=410, ip=10.244.7.203) Evaluation complete:[1954/1954]
(RayWorker pid=410, ip=10.244.7.203) ACC = 0.7712157964706421
(RayWorker pid=410, ip=10.244.7.203) AUC = 0.7507861852645874
(RayWorker pid=333, ip=10.244.10.244) Evaluation complete:[1954/1954]
(RayWorker pid=333, ip=10.244.10.244) ACC = 0.7706922292709351
(RayWorker pid=333, ip=10.244.10.244) AUC = 0.7526524066925049
```
