# IPEX-LLM-Serving CPU Image: Build and Usage Guide

This document provides instructions for building and using the `IPEX-LLM-serving` CPU Docker image, including model inference, serving, and benchmarking functionalities.


---

## 1. Build the Image  

To build the `ipex-llm-serving-cpu` Docker image, run the following command:  

```bash
docker build \
  --build-arg http_proxy=.. \
  --build-arg https_proxy=.. \
  --build-arg no_proxy=.. \
  --rm --no-cache -t intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT .
```

---

## 2. Run the Container  

Before running `chat.py` or using serving functionalities, start the container using the following command.  

### **Step 1: Download the Model (Optional)**  

If using a local model, download it to your host machine and bind the directory to the container when launching it.  

```bash
export MODEL_PATH=/home/llm/models  # Change this to your model directory
```

This ensures the container has access to the necessary models.  

---

### **Step 2: Start the Container**  

Use the following command to start the container:  

```bash
export DOCKER_IMAGE=intelanalytics/ipex-llm-serving-cpu:2.2.0-SNAPSHOT

sudo docker run -itd \
        --net=host \  # Use host networking for performance
        --cpuset-cpus="0-47" \  # Limit the container to specific CPU cores
        --cpuset-mems="0" \  # Bind the container to NUMA node 0 for memory locality
        --memory="32G" \  # Limit memory usage to 32GB
        --shm-size="16g" \  # Set shared memory size to 16GB (useful for large models)
        --name=CONTAINER_NAME \
        -v $MODEL_PATH:/llm/models/ \  # Mount the model directory
        $DOCKER_IMAGE
```

### **Step 3: Access the Running Container**  

Once the container is started, you can access it using:  

```bash
sudo docker exec -it CONTAINER_NAME bash
```

---

## 3. Using `chat.py` for Inference  

The `chat.py` script is used for model inference. It is located under the `/llm` directory inside the container.  

### Steps:  

1. **Run `chat.py` for inference** inside the container:  

   ```bash
   cd /llm
   python chat.py --model-path /llm/models/MODEL_NAME
   ```

   Replace `MODEL_NAME` with the name of your model.  

---

## 4. Serving with IPEX-LLM  

The container supports multiple serving engines.  

### 4.1 Serving with FastChat Engine  

To run FastChat-serving using `IPEX-LLM` as the backend, refer to this [document](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/src/ipex_llm/serving/fastchat).  

---

### 4.2 Serving with vLLM Engine  

To use **vLLM** with `IPEX-LLM` as the backend, refer to the [vLLM Serving Guide](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md).  

The following example files are included in the `/llm/` directory inside the container:  

- `vllm_offline_inference.py`: Used for vLLM offline inference example.  
- `benchmark_vllm_throughput.py`: Used for throughput benchmarking.  
- `payload-1024.lua`: Used for testing requests per second with a 1k-128 request pattern.  
- `start-vllm-service.sh`: Template script for starting the vLLM service.  

---

## 5. Benchmarks  

### 5.1 Online Benchmark through API Server  

To benchmark the API Server and estimate transactions per second (TPS), first start the service as per the instructions in the [vLLM Serving Guide](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/GPU/vLLM-Serving/README.md#service).  

Then, follow these steps:  

1. **Modify the `payload-1024.lua` file** to ensure the `"model"` attribute is correctly set.  
2. **Run the benchmark using `wrk`**:  

   ```bash
   cd /llm
   # You can adjust -t and -c to control concurrency.
   wrk -t4 -c4 -d15m -s payload-1024.lua http://localhost:8000/v1/completions --timeout 1h
   ```

---

### 5.2 Offline Benchmark through `benchmark_vllm_throughput.py`  

1. **Download the test dataset**:  

   ```bash
   wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
   ```

2. **Run the benchmark script**:  

   ```bash
   cd /llm/

   export MODEL="YOUR_MODEL"

   # You can change load-in-low-bit from values in [sym_int4, fp8, fp16]
   python3 /llm/benchmark_vllm_throughput.py \
       --backend vllm \
       --dataset /llm/ShareGPT_V3_unfiltered_cleaned_split.json \
       --model $MODEL \
       --num-prompts 1000 \
       --seed 42 \
       --trust-remote-code \
       --enforce-eager \
       --dtype bfloat16 \
       --device cpu \
       --load-in-low-bit sym_int4
   ```

---