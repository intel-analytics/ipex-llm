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
#


import os
import time
import torch
import argparse
from ipex_llm.transformers.npu_model import EmbeddingModel
from transformers.utils import logging

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="maidalun1020/bce-embedding-base_v1",
        help="The huggingface repo id for the bce-embedding model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the lowbit model folder, leave blank if you do not want to save. \
            If path not exists, lowbit model will be saved there. \
            Else, lowbit model will be loaded.",
    )
    parser.add_argument('--prompt', type=str, default="'sentence_0', 'sentence_1'",
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)
    parser.add_argument("--intra-pp", type=int, default=2)
    parser.add_argument("--inter-pp", type=int, default=2)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    model = EmbeddingModel(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="eager",
        optimize_model=True,
        max_context_len=args.max_context_len,
        max_prompt_len=args.max_prompt_len,
        intra_pp=args.intra_pp,
        inter_pp=args.inter_pp,
        transpose_value_cache=not args.disable_transpose_value_cache,
    )

    # list of sentences
    split_items = args.prompt.split(',')
    sentences = [item.strip().strip("'") for item in split_items]

    # extract embeddings
    st = time.time()
    embeddings = model.encode(sentences)
    end = time.time()
    print(f'Inference time: {end-st} s')
    print(embeddings)