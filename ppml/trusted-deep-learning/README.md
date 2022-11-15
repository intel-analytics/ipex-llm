# Gramine
SGX-based Trusted Deep Learning allows the user to run end-to-end deep-learning training application in a secure environment.

The following sections will show how to run a small demo using our currently-developed image.

*Please mind the IP and file path settings. They should be changed to the IP/path of your own sgx server on which you are running the programs.*

## Before Running code
### 1. Build Docker Images

**Tip:** if you want to skip building the custom image, you can use our public image `intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.2.0-SNAPSHOT` for a quick start, which is provided for a demo purpose. Do not use it in production.

#### 1.1 Build BigDL Base Image

The bigdl base image is a public one that does not contain any secrets. You will use the base image to get your own custom image in the following steps. 

Please be noted that the `intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-base:2.2.0-SNAPSHOT` image relies on the `intelanalytics/bigdl-ppml-gramine-base:2.2.0-SNAPSHOT` image.  

For the instructions on how to build the `gramine-base` image, check `ppml/base/README.md` in our repository.  Another option is to use our public image `intelanalytics/bigdl-ppml-gramine-base:2.2.0-SNAPSHOT` for a quick start.

Before running the following command, please modify the paths in `../base/build-docker-image.sh`. Then build the docker image with the following command.

```bash
# Assuming you are in ppml/trusted-deep-learning/base directory 
# configure parameters in build-docker-image.sh please
./build-docker-image.sh
```
#### 1.2 Build Customer Image

First, You need to generate your enclave key using the command below, and keep it safe for future remote attestations and to start SGX enclaves more securely.

It will generate a file `enclave-key.pem` in `ppml/trusted-deep-learning/ref` directory, which will be your enclave key. To store the key elsewhere, modify the outputted file path.

```bash
# Assuming you are in ppml/trusted-deep-learning/ref directory
openssl genrsa -3 -out enclave-key.pem 3072
```

Then, use the `enclave-key.pem` and the `intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-base:2.2.0-SNAPSHOT` image to build your own custom image. In the process, SGX MREnclave will be made and signed without saving the sensitive enclave key inside the final image, which is safer.


Before running the following command, please modify the paths in `./build-custom-image.sh`. Then build the docker image with the following command.

```bash
# under ppml/trusted-deep-learning/ref dir
# modify custom parameters in build-custom-image.sh
./build-custom-image.sh
```

The docker build console will also output `mr_enclave` and `mr_signer` like below, which are hash values and used to  register your MREnclave in the following.

````bash
......
[INFO] Use the below hash values of mr_enclave and mr_signer to register enclave:
mr_enclave       : c7a8a42af......
mr_signer        : 6f0627955......
````

### 2. Prepare TLS keys

To enable TLS in GLOO backend, we need to setup the following three parameters:

1. GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY
2. GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT
3. GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE


#### 2.1 Prepare the root CA file


You can use your test CA file in your organization, or create a new CA file for test purpose using the following command:
 
```bash
# Generate the CA pkey
openssl genrsa -des3 -out myCA.key 2048 # Use of pass-phrase is recommended
# Generate the CA permission
openssl req -x509 -new -nodes -key myCA.key -sha256 -days 1825 -out myCA.pem
```

Fill in the required items and you will get your CA file at `./myCA.pem`

#### 2.2 Prepare the TLS_PKEY

This step is simply, just run the following steps to acquire your private key

```bash
openssl genrsa -out test.key 2048
```

#### 2.3 Prepare the TLS_CERTIFICATE

First we use our private key to generate a Certificate Sign Request:

```bash
openssl req -new -key test.key -out test.csr
```

Finally, we use our CA private key and CA file to sign our CSR:
```bash
openssl x509 -req -in test.csr -CA myCA.pem -CAkey myCA.key \
-CAcreateserial -out test.crt -days 825 -sha256 
```

Now you should have all files required for TLS encryption, including `myCA.pem`, `test.key` and `test.crt`.




### Demo

*WARNING: We are currently actively developing our images, which indicate that the ENTRYPOINT of the docker image may be changed accordingly.  We will do our best to update our documentation in time.*

We have included a file named `mnist.py` in our `intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.2.0-SNAPSHOT` image for test purpose.  In the following sections, we will show how to run distributed PyTorch training in nodes with SGX enabled.

Run the following script on nodes with SGX enabled:

To run the following bash scripts, set the following parameters:
1. `CERTS_PATH`
2. `MASTER_ADDR`
3. `http_proxy`
4. `https_proxy`
5. `no_proxy`
6. TLS related settings if needed

#### On node one

```bash
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.2.0-SNAPSHOT
export CERTS_PATH="your_certs_path"
sudo docker run -itd \
        --net=host \
        --name=node_one \
        --cpuset-cpus="20-24" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $CERTS_PATH:/ppml/certs \
        -e SGX_ENABLED=true \
        -e SGX_LOG_LEVEL=error \
        -e WORLD_SIZE=3 \
        -e RANK=0 \
        -e MASTER_PORT=29500 \
        -e MASTER_ADDR="your_master_addr" \
        -e GLOO_DEVICE_TRANSPORT="TCP_TLS" \
        -e GLOO_TCP_IFACE="ens259f0" \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/ppml/certs/test.key \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/ppml/certs/test.crt \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/ppml/certs/myCA.pem \
        -e http_proxy="your_http_proxy" \
        -e https_proxy="your_https_proxy" \
        -e no_proxy="your_no_proxy" \
        $DOCKER_IMAGE python3 mnist.py --epoch 10 --no-cuda --seed 42 --save-model

docker logs -f node_one
```

#### On node two
```bash
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.2.0-SNAPSHOT
export CERTS_PATH="your_certs_path"
sudo docker run -itd \
        --net=host \
        --name=node_two \
        --cpuset-cpus="20-24" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $CERTS_PATH:/ppml/certs \
        -e SGX_ENABLED=true \
        -e SGX_LOG_LEVEL=error \
        -e WORLD_SIZE=3 \
        -e RANK=1 \
        -e MASTER_PORT=29500 \
        -e MASTER_ADDR="your_master_addr" \
        -e GLOO_DEVICE_TRANSPORT="TCP_TLS" \
        -e GLOO_TCP_IFACE="ens259f0" \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/ppml/certs/test.key \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/ppml/certs/test.crt \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/ppml/certs/myCA.pem \
        -e http_proxy="your_http_proxy" \
        -e https_proxy="your_https_proxy" \
        -e no_proxy="your_no_proxy" \
        $DOCKER_IMAGE python3 mnist.py --epoch 10 --no-cuda --seed 42 --save-model

docker logs -f node_two
```


#### On node three

```bash
export DOCKER_IMAGE=intelanalytics/bigdl-ppml-trusted-deep-learning-gramine-ref:2.2.0-SNAPSHOT
export CERTS_PATH="your_certs_path"
sudo docker run -itd \
        --net=host \
        --name=node_three \
        --cpuset-cpus="20-24" \
        --oom-kill-disable \
        --device=/dev/sgx/enclave \
        --device=/dev/sgx/provision \
        -v /var/run/aesmd/aesm.socket:/var/run/aesmd/aesm.socket \
        -v $CERTS_PATH:/ppml/certs \
        -e SGX_ENABLED=true \
        -e SGX_LOG_LEVEL=error \
        -e WORLD_SIZE=3 \
        -e RANK=2 \
        -e MASTER_PORT=29500 \
        -e MASTER_ADDR="your_master_addr" \
        -e GLOO_DEVICE_TRANSPORT="TCP_TLS" \
        -e GLOO_TCP_IFACE="ens259f0" \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_PKEY=/ppml/certs/test.key \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CERT=/ppml/certs/test.crt \
        -e GLOO_DEVICE_TRANSPORT_TCP_TLS_CA_FILE=/ppml/certs/myCA.pem \
        -e http_proxy="your_http_proxy" \
        -e https_proxy="your_https_proxy" \
        -e no_proxy="your_no_proxy" \
        $DOCKER_IMAGE python3 mnist.py --epoch 10 --no-cuda --seed 42 --save-model

docker logs -f node_three
```


#### Watch the result

After you have booted the required worker containers, you can watch the training process by using the following command:

```bash
docker logs -f node_one
```

It should be similar to the following:

```log
2022-10-20T02:00:33Z INFO     Train Epoch: 10 [23680/60000 (39%)]       loss=0.3618
2022-10-20T02:00:33Z INFO     Train Epoch: 10 [24320/60000 (41%)]       loss=0.3449
2022-10-20T02:00:33Z INFO     Train Epoch: 10 [24960/60000 (42%)]       loss=0.2862
2022-10-20T02:00:34Z INFO     Train Epoch: 10 [25600/60000 (43%)]       loss=0.2990
2022-10-20T02:00:34Z INFO     Train Epoch: 10 [26240/60000 (44%)]       loss=0.5047
2022-10-20T02:00:34Z INFO     Train Epoch: 10 [26880/60000 (45%)]       loss=0.3102
2022-10-20T02:00:35Z INFO     Train Epoch: 10 [27520/60000 (46%)]       loss=0.1734
2022-10-20T02:00:35Z INFO     Train Epoch: 10 [28160/60000 (47%)]       loss=0.2250
2022-10-20T02:00:35Z INFO     Train Epoch: 10 [28800/60000 (48%)]       loss=0.3896
2022-10-20T02:00:36Z INFO     Train Epoch: 10 [29440/60000 (49%)]       loss=0.4225
2022-10-20T02:00:36Z INFO     Train Epoch: 10 [30080/60000 (50%)]       loss=0.3704
2022-10-20T02:00:36Z INFO     Train Epoch: 10 [30720/60000 (51%)]       loss=0.3256
2022-10-20T02:00:37Z INFO     Train Epoch: 10 [31360/60000 (52%)]       loss=0.2525
2022-10-20T02:00:37Z INFO     Train Epoch: 10 [32000/60000 (53%)]       loss=0.2592
2022-10-20T02:00:37Z INFO     Train Epoch: 10 [32640/60000 (54%)]       loss=0.2287
2022-10-20T02:00:38Z INFO     Train Epoch: 10 [33280/60000 (55%)]       loss=0.2164
2022-10-20T02:00:38Z INFO     Train Epoch: 10 [33920/60000 (57%)]       loss=0.2710
2022-10-20T02:00:38Z INFO     Train Epoch: 10 [34560/60000 (58%)]       loss=0.2571
2022-10-20T02:00:39Z INFO     Train Epoch: 10 [35200/60000 (59%)]       loss=0.1178
2022-10-20T02:00:39Z INFO     Train Epoch: 10 [35840/60000 (60%)]       loss=0.2550
2022-10-20T02:00:39Z INFO     Train Epoch: 10 [36480/60000 (61%)]       loss=0.5370
2022-10-20T02:00:40Z INFO     Train Epoch: 10 [37120/60000 (62%)]       loss=0.4263
2022-10-20T02:00:40Z INFO     Train Epoch: 10 [37760/60000 (63%)]       loss=0.5719
2022-10-20T02:00:41Z INFO     Train Epoch: 10 [38400/60000 (64%)]       loss=0.3927
2022-10-20T02:00:41Z INFO     Train Epoch: 10 [39040/60000 (65%)]       loss=0.5021
2022-10-20T02:00:41Z INFO     Train Epoch: 10 [39680/60000 (66%)]       loss=0.2474
2022-10-20T02:00:41Z INFO     Train Epoch: 10 [40320/60000 (67%)]       loss=0.1810
2022-10-20T02:00:42Z INFO     Train Epoch: 10 [40960/60000 (68%)]       loss=0.2563
2022-10-20T02:00:42Z INFO     Train Epoch: 10 [41600/60000 (69%)]       loss=0.3363
2022-10-20T02:00:42Z INFO     Train Epoch: 10 [42240/60000 (70%)]       loss=0.2499
2022-10-20T02:00:43Z INFO     Train Epoch: 10 [42880/60000 (71%)]       loss=0.3590
2022-10-20T02:00:43Z INFO     Train Epoch: 10 [43520/60000 (72%)]       loss=0.2877
2022-10-20T02:00:43Z INFO     Train Epoch: 10 [44160/60000 (74%)]       loss=0.4449
2022-10-20T02:00:44Z INFO     Train Epoch: 10 [44800/60000 (75%)]       loss=0.1696
2022-10-20T02:00:44Z INFO     Train Epoch: 10 [45440/60000 (76%)]       loss=0.1592
2022-10-20T02:00:44Z INFO     Train Epoch: 10 [46080/60000 (77%)]       loss=0.3730
2022-10-20T02:00:45Z INFO     Train Epoch: 10 [46720/60000 (78%)]       loss=0.3739
2022-10-20T02:00:45Z INFO     Train Epoch: 10 [47360/60000 (79%)]       loss=0.3429
2022-10-20T02:00:45Z INFO     Train Epoch: 10 [48000/60000 (80%)]       loss=0.4184
2022-10-20T02:00:45Z INFO     Train Epoch: 10 [48640/60000 (81%)]       loss=0.1960
2022-10-20T02:00:46Z INFO     Train Epoch: 10 [49280/60000 (82%)]       loss=0.2144
2022-10-20T02:00:46Z INFO     Train Epoch: 10 [49920/60000 (83%)]       loss=0.2799
2022-10-20T02:00:46Z INFO     Train Epoch: 10 [50560/60000 (84%)]       loss=0.2553
2022-10-20T02:00:47Z INFO     Train Epoch: 10 [51200/60000 (85%)]       loss=0.2735
2022-10-20T02:00:47Z INFO     Train Epoch: 10 [51840/60000 (86%)]       loss=0.3195
2022-10-20T02:00:47Z INFO     Train Epoch: 10 [52480/60000 (87%)]       loss=0.4039
2022-10-20T02:00:48Z INFO     Train Epoch: 10 [53120/60000 (88%)]       loss=0.2460
2022-10-20T02:00:48Z INFO     Train Epoch: 10 [53760/60000 (90%)]       loss=0.3027
2022-10-20T02:00:48Z INFO     Train Epoch: 10 [54400/60000 (91%)]       loss=0.2507
2022-10-20T02:00:49Z INFO     Train Epoch: 10 [55040/60000 (92%)]       loss=0.2123
2022-10-20T02:00:49Z INFO     Train Epoch: 10 [55680/60000 (93%)]       loss=0.3268
2022-10-20T02:00:50Z INFO     Train Epoch: 10 [56320/60000 (94%)]       loss=0.2732
2022-10-20T02:00:50Z INFO     Train Epoch: 10 [56960/60000 (95%)]       loss=0.2821
2022-10-20T02:00:50Z INFO     Train Epoch: 10 [57600/60000 (96%)]       loss=0.2649
2022-10-20T02:00:51Z INFO     Train Epoch: 10 [58240/60000 (97%)]       loss=0.4038
2022-10-20T02:00:51Z INFO     Train Epoch: 10 [58880/60000 (98%)]       loss=0.2187
2022-10-20T02:00:51Z INFO     Train Epoch: 10 [59520/60000 (99%)]       loss=0.3429
```


### Running distributed training using Kubernetes

If Kubernetes is used to boot multiple PyTorch training processes, then please pay attention to the following cluster configs, which may have a huge impact on training speed.


#### Node CPU manager

> TL;DR Set the CPU Management policy to `static` on Kubernetes nodes where you want to run Distributed PyTorch training.


According to the [feature-highlight](https://kubernetes.io/blog/2018/07/24/feature-highlight-cpu-manager/?spm=a2c65.11461447.0.0.14399444yu2dvp#sounds-good-but-does-the-cpu-manager-help-me), the CPU manager might help workloads with the following characteristics:
1. Sensitive to CPU throttling effects.
2. Sensitive to context switches.
3. Sensitive to processor cache misses.
4. Benefits from sharing a processor resources (e.g., data and instruction caches).
5. Sensitive to cross-socket memory traffic.
6. Sensitive or requires hyperthreads from the same physical CPU core.

The PyTorch distributed training is a CPU-intensive workloads that is sensitive to context switches when SGX is enabled.  By setting the CPU management policy to `static`, the PyTorch training process can exclusively use the provided cores, which largely reduce the overhead of context-switching and TLB flushing.

Under our cluster, the training is around eight times faster on nodes with static CPU management policy than nodes with no policy set.


#### NUMA node setting
>TL;DR Set the Cpu Topology Management Policy to `best-effort` or `single-numa-node`


Generally speaking, all deep learning workloads, training or inference, get better performance without accessing hardware resources across NUMA nodes. Under our cluster, the training is around four times faster on logical cores in the same NUMA node than logical cores across NUMA node.

PyTorch also recommends to utilize NUMA when do PyTorch training, more details can be found [here](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#utilize-non-uniform-memory-access-numa-controls).



#### Hyper-threading setting
>TL;DR The use of Hyper-threading can increase the computation throughput of the cluster.  However, it may have negative effects on training speed of distributed training.

In native mode (with SGX disabled), the use of hyper-threading may increase the training time because two logical cores reside in the same physical core may not be able to execute fully in parallel.

In SGX mode, there is a problem when using Hyper-threading combined with Gramine, which leads to the result that only one of the hyper-threads in the physical core can be fully utilized.  We are currently investigating this issue.  In this case, the use of hyper-threads may bring additional overheads to the distributed training.
