# LoRA Fine-Tuning on ChatGLM3-6B with IPEX-LLM

This example ports [ChatGLM3-6B lora_finetune](https://github.com/THUDM/ChatGLM3/blob/main/finetune_demo/lora_finetune.ipynb) demo to IPEX-LLM on [Intel Arc GPU](../../README.md).

### 1. Install

```bash
conda create -n llm python=3.11
conda activate llm
pip install "jieba>=0.42.1"
pip install "ruamel_yaml>=0.18.6"
pip install "rouge_chinese>=1.0.3"
pip install "jupyter>=1.0.0"
pip install "datasets>=2.18.0"
pip install "peft>=0.10.0"
pip install typer
pip install sentencepiece
pip install nltk
pip install "numpy<2.0.0"
pip install "deepspeed==0.13.1"
pip install "mpi4py>=3.1.5"
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install oneccl_bind_pt==2.1.100 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

### 2. Configures OneAPI Environment Variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. LoRA Fine-Tune on ChatGLM3-6B

First, as for the dataset, you have two options:

1. `AdvertiseGen`: please now get it from [Google Drive](https://drive.google.com/file/d/13_vf0xRTQsyneRKdD1bZIr93vBGOczrk/view?usp=sharing) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/b3f119a008264b1cabd1/?dl=1), and unzip it in the current directory. Then, process the dataset with the below script:

```bash
python process_advertise_gen_dataset.py
```

Then, './AdvertiseGen' will be converted to './AdvertiseGen_fix'. Now, we have prepared the dataset, and are going to start LoRA fine-tuning on ChatGLM3-6B.

2. `Alapca`: We also support [yahma/alpaca-cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned) that contains generated instructions and demonstrations. It does not require preprocessing, and please directy run the following script.

#### 3.1. Fine-Tune with a Single Arc Card

1. For `AdvertiseGen`, start the fine-tuning by:

```bash
bash lora_finetuning_chatglm3_6b_on_advertise_gen_with_1_arc_card.sh
```

2. For `Alpaca`, start the fine-tuning by:

```bash
bash lora_finetuning_chatglm3_6b_on_alpaca_with_1_arc_card.sh
```

Then, you will get output are as below:

```bash
2024-06-27 13:47:02,680 - root - INFO - intel_extension_for_pytorch auto imported
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.47it/s]
2024-06-27 13:47:03,794 - ipex_llm.transformers.utils - INFO - Converting the current model to bf16 format......
[2024-06-27 13:47:04,105] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to xpu (auto detect)
trainable params: 487,424 || all params: 6,244,071,424 || trainable%: 0.0078
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): ChatGLMForConditionalGeneration(
      (transformer): ChatGLMModel(
        (embedding): Embedding(
          (word_embeddings): Embedding(65024, 4096)
        )
        (rotary_pos_emb): RotaryEmbedding()
        (encoder): GLMTransformer(
          (layers): ModuleList(
            (0-27): 28 x GLMBlock(
              (input_layernorm): RMSNorm()
              (self_attention): SelfAttention(
                (query_key_value): LoraLowBitLinear(
                  (base_layer): BF16Linear(in_features=4096, out_features=4608, bias=True)
                  (lora_dropout): ModuleDict(
                    (default): Dropout(p=0.1, inplace=False)
                  )
                  (lora_A): ModuleDict(
                    (default): Linear(in_features=4096, out_features=2, bias=False)
                  )
                  (lora_B): ModuleDict(
                    (default): Linear(in_features=2, out_features=4608, bias=False)
                  )
                  (lora_embedding_A): ParameterDict()
                  (lora_embedding_B): ParameterDict()
                  (qa_pool): Identity()
                )
                (core_attention): CoreAttention(
                  (attention_dropout): Dropout(p=0.0, inplace=False)
                )
                (dense): BF16Linear(in_features=4096, out_features=4096, bias=False)
              )
              (post_attention_layernorm): RMSNorm()
              (mlp): MLP(
                (dense_h_to_4h): BF16Linear(in_features=4096, out_features=27392, bias=False)
                (dense_4h_to_h): BF16Linear(in_features=13696, out_features=4096, bias=False)
              )
            )
          )
          (final_layernorm): RMSNorm()
        )
        (output_layer): BF16Linear(in_features=4096, out_features=65024, bias=False)
      )
    )
  )
)
--> Model

--> model has 0.487424M params

train_dataset: Dataset({
    features: ['input_ids', 'labels'],
    num_rows: 114599
})
val_dataset: Dataset({
    features: ['input_ids', 'output_ids'],
    num_rows: 1070
})
test_dataset: Dataset({
    features: ['input_ids', 'output_ids'],
    num_rows: 1070
})
--> Sanity check
           '[gMASK]': 64790 -> -100
               'sop': 64792 -> -100
          '<|user|>': 64795 -> -100
                  '': 30910 -> -100
                '\n': 13 -> -100
......

# Here it takes time to finish the whole fine-tuning

......

Training completed. Do not forget to share your model on huggingface.co/models =)


{'train_runtime': xxxx.xxxx, 'train_samples_per_second': x.xxx, 'train_steps_per_second': x.xxx, 'train_loss': xx.xx, 'epoch': x.xx}
100%|████████████████████████████████████████████████████████████████████████████████████████████| 3000/3000 [xx:xx<00:00,  x.xxit/s]
***** Running Prediction *****
  Num examples = 1070
  Batch size = 4
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 268/268 [xx:xx<00:00,  x.xxs/it]
```

#### 3.2. Fine-Tune with 2 Arc Cards

Start the data-parallel fine-tuning on 2 Intel Arc XPU cards by:

1. `AdvertiseGen` dataset:

```bash
bash lora_finetuning_chatglm3_6b_on_advertise_gen_with_2_arc_cards.sh
```

2. `Alpaca` dataset:

```bash
bash lora_finetuning_chatglm3_6b_on_alpaca_with_2_arc_cards.sh
```
