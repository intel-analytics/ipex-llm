#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This is modified from https://github.com/intel-sandbox/customer-ai-test-code/blob/main/convert-model-textgen-to-classfication.py
#
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model_path', type=str, help='an string for the device')
args = parser.parse_args()
model_path = args.model_path

dtype=torch.bfloat16
num_labels = 5
model_name=model_path
save_directory = model_name + "-classification"

# Initialize the tokenizer 
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.save_pretrained(save_directory)


model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, pad_token_id=tokenizer.eos_token_id,)
config = AutoConfig.from_pretrained(model_name)
print("text gen model")
print(model)
print(config)


model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, torch_dtype=dtype)
save_directory = model_name + "-classification"
model.save_pretrained(save_directory)    


model = AutoModelForSequenceClassification.from_pretrained(save_directory, torch_dtype=dtype, pad_token_id=tokenizer.eos_token_id)
config = AutoConfig.from_pretrained(save_directory)
print("text classification model")
print(model)
print(config)
