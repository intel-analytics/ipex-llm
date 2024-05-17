# LlaMA

In this directory, you will find examples on how you could apply IPEX-LLM INT4 optimizations on general pytorch models, for example Meta Llama models. **Different from what [Huggingface LlaMA2](../llama2/) example demonstrated, This example directly brings the optimizations of IPEX-LLM to the official LLaMA implementation of which the code style is more flexible.** For illustration purposes, we utilize the [Llama2-7b-Chat](https://ai.meta.com/llama/) as a reference LlaMA model.

## Requirements
To run these examples with IPEX-LLM, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Generating text using a pretrained Llama model
In the example [example_chat_completion.py](./example_chat_completion.py), we show a basic use case for a Llama model to engage in a conversation with an AI assistant using `chat_completion` API, with IPEX-LLM INT4 optimizations. The process for [example_text_completion.py](./example_text_completion.py) is similar.
### 1. Install
We suggest using conda to manage environment:

On Linux:

```bash
conda create -n llm python=3.11
conda activate llm

# Install meta-llama repository
git clone https://github.com/facebookresearch/llama.git
cd llama/
git apply < ../cpu.patch # apply cpu version patch
pip install -e .

# install the latest ipex-llm nightly build with 'all' option
pip install --pre --upgrade ipex-llm[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

On Windows:

```cmd
conda create -n llm python=3.11
conda activate llm

git clone https://github.com/facebookresearch/llama.git
cd llama/
git apply < ../cpu.patch
pip install -e .

pip install --pre --upgrade ipex-llm[all]
```

### 2. Run
Follow the instruction [here](https://github.com/facebookresearch/llama#download) to download the model weights and tokenizer.
```
torchrun --nproc-per-node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 64 --max_batch_size 1 --backend cpu
```

Arguments info:
- `--ckpt_dir` (str): The directory containing checkpoint files for the pretrained model.
- `--tokenizer_path` (str): The path to the tokenizer model used for text encoding/decoding.
- `--temperature` (float, optional): The temperature value for controlling randomness in generation.
    Defaults to 0.6.
- `--top_p` (float, optional): The top-p sampling parameter for controlling diversity in generation.
    Defaults to 0.9.
- `--max_seq_len` (int, optional): The maximum sequence length for input prompts. Defaults to 128.
- `--max_gen_len` (int, optional): The maximum length of generated sequences. Defaults to 64.
- `--max_batch_size` (int, optional): The maximum batch size for generating sequences. Defaults to 4.
- `--backend` (str): The device backend for computing. Defaults to `cpu`.

> Please select the appropriate size of the Llama model based on the capabilities of your machine.


#### 2.1 Client
On client Windows machine, it is recommended to run directly with full utilization of all cores:
```cmd
torchrun --nproc-per-node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 64 --max_batch_size 1 --backend cpu
```

#### 2.2 Server
For optimal performance on server, it is recommended to set several environment variables (refer to [here](../README.md#best-known-configuration-on-linux) for more information), and run the example with all the physical cores of a single socket.

E.g. on Linux,
```bash
# set IPEX-Nano env variables
source ipex-nano-init

# e.g. for a server with 48 cores per socket
export OMP_NUM_THREADS=48
numactl -C 0-47 -m 0 torchrun --nproc-per-node 1 example_chat_completion.py --ckpt_dir llama-2-7b-chat/ --tokenizer_path tokenizer.model --max_seq_len 64 --max_batch_size 1 --backend cpu
```

#### 2.3 Sample Output
#### [Llama2-7b-Chat](https://ai.meta.com/llama/)

```log
2023-10-08 13:49:11,107 - INFO - Added key: store_based_barrier_key:1 to store for rank: 0
2023-10-08 13:49:11,108 - INFO - Rank 0: Completed store-based barrier for key:store_based_barrier_key:1 with 1 nodes.
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
2023-10-08 13:49:11,130 - INFO - Added key: store_based_barrier_key:2 to store for rank: 0
2023-10-08 13:49:11,130 - INFO - Rank 0: Completed store-based barrier for key:store_based_barrier_key:2 with 1 nodes.
2023-10-08 13:49:11,131 - INFO - Added key: store_based_barrier_key:3 to store for rank: 0
2023-10-08 13:49:11,131 - INFO - Rank 0: Completed store-based barrier for key:store_based_barrier_key:3 with 1 nodes.
2023-10-08 13:49:11,132 - INFO - Added key: store_based_barrier_key:4 to store for rank: 0
2023-10-08 13:49:11,132 - INFO - Rank 0: Completed store-based barrier for key:store_based_barrier_key:4 with 1 nodes.
2023-10-08 13:49:19,108 - INFO - Reloaded SentencePiece model from /disk1/changmin/Llama-2-7b-chat/tokenizer.model
2023-10-08 13:49:19,108 - INFO - #words: 32000 - BOS ID: 1 - EOS ID: 2
Loaded in 54.41 seconds
2023-10-08 13:50:09,600 - INFO - Only HuggingFace Transformers models are currently supported for further optimizations
User: what is the recipe of mayonnaise?

> Assistant:  Mayonnaise is a thick, creamy condiment made from a mixture of egg yolks, oil, and vinegar or lemon juice. Unterscheidung of mayonnaise involves the use of an emuls

==================================
```
