import torch
import os
import transformers
from transformers import LlamaTokenizer

from peft import LoraConfig
import intel_extension_for_pytorch as ipex
from bigdl.llm.transformers.qlora import get_peft_model, prepare_model_for_kbit_training
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse

if __name__ == "__main__":

    model_path = "/model" if os.path.exists("/model") else "meta-llama/Llama-2-7b-hf"
    dataset_path = "/data" if os.path.exists("/data") else "Abirate/english_quotes"
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data = load_dataset(dataset_path)
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                load_in_4bit=True,
                                                optimize_model=False,
                                                modules_to_not_convert=["lm_head"],)
    model = model.to('xpu')
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps= 1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-4,
            fp16=False, # fp16 is not supported yet
            logging_steps=20,
            output_dir="outputs",
            optim="adamw_hf", # paged_adamw_8bit is not supported yet
            # gradient_checkpointing=True, # can further reduce memory but slower
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
