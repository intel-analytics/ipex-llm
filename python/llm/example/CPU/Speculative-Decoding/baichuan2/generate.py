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

import torch
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import LlamaTokenizer, AutoTokenizer
import argparse
import time
import numpy as np

torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
# here the prompt tuning refers to https://huggingface.co/georgesung/llama2_7b_chat_uncensored#prompt-style
BAICHUAN_PROMPT_FORMAT = "<human>{prompt} <bot>"

long_input = """æŠ˜çº¸çš„è¿‡ç¨‹çœ‹ä¼¼ç®€å•ï¼Œå…¶å®žæƒ³è¦åšå¥½ï¼Œè¿˜æ˜¯éœ€è¦ä¸€å¥—å¾ˆå¤æ‚çš„å·¥è‰ºã€‚ä»¥æŠ˜ä¸€æ”¯çŽ«ç‘°èŠ±ä¸ºä¾‹ï¼Œæˆ‘ä»¬å¯ä»¥å°†æ•´ä¸ªæŠ˜çº¸è¿‡ç¨‹åˆ†æˆä¸‰ä¸ªé˜¶æ®µï¼Œå³ï¼šåˆ›å»ºæ …æ ¼æŠ˜ç—•ï¼Œåˆ¶ä½œç«‹ä½“åŸº
åº§ï¼Œå®ŒæˆèŠ±ç“£ä¿®é¥°ã€‚é¦–å…ˆæ˜¯åˆ›å»ºæ …æ ¼æŠ˜ç—•ï¼šè¿™ä¸€æ­¥æœ‰ç‚¹åƒ>æˆ‘ä»¬æŠ˜åƒçº¸é¹¤çš„ç¬¬ä¸€æ­¥ï¼Œå³é€šè¿‡å¯¹ç§°å·žä¾æ¬¡å¯¹æŠ˜ï¼Œç„¶åŽæŒ‰ç…§é•¿å’Œå®½ä¸¤ä¸ªç»´åº¦ï¼Œä¾æ¬¡è¿›è¡Œå¤šç­‰åˆ†çš„å‡åŒ€æŠ˜å ï¼›>æœ€ç»ˆåœ¨ä¸¤ä¸ªæ–¹å‘ä¸Šçš„æŠ˜ç—•ä¼šäº¤ç»‡æˆä¸€å¥—å®Œæ•´å‡åŒ€çš„å°æ–¹æ ¼æ‹¼æŽ¥å›¾æ¡ˆï¼›è¿™äº›å°æ–¹æ ¼å°±ç»„æˆäº†ç±»ä¼¼äºŒç»´åæ ‡ç³»çš„å‚è€ƒç³»ç»Ÿï¼Œä½¿å¾—æˆ‘ä»¬åœ¨>è¯¥å¹³é¢ä¸Šï¼Œé€šè¿‡ç»„åˆä¸´è¿‘æŠ˜ç—•çš„æ–¹>å¼ä»ŽäºŒç»´å°æ–¹æ ¼ä¸ŠæŠ˜å å‡ºä¸‰ç»´çš„é«˜å°æˆ–å‡¹é™·ï¼Œä»¥ä¾¿äºŽæŽ¥ä¸‹æ¥çš„å‡ åº§åˆ¶ä½œè¿‡ç¨‹ã€‚éœ€è¦æ³¨æ„çš„æ˜¯ï¼Œåœ¨å»ºç«‹æ …æ ¼æŠ˜ç—•çš„è¿‡ç¨‹ä¸­ï¼Œå¯èƒ½ä¼šå‡ºçŽ°æŠ˜å ä¸å¯¹æˆçš„æƒ…å†µï¼Œè¿™ç§é”™è¯¯æ‰€å¸¦
æ¥çš„åŽæžœå¯èƒ½æ˜¯å¾ˆä¸¥é‡çš„ï¼Œå°±åƒæ˜¯è´>è¶æ•ˆåº”ï¼Œä¸€å¼€å§‹åªæ˜¯æ¯«åŽ˜ä¹‹å·®ï¼Œæœ€åŽå¯èƒ½å°±æ˜¯å¤©å£¤ä¹‹åˆ«ã€‚ç„¶åŽæ˜¯åˆ¶ä½œç«‹ä½“åŸºåº§ï¼šåœ¨è¿™ä¸€æ­¥ï¼Œæˆ‘ä»¬éœ€è¦åŸºäºŽæ …æ ¼æŠ˜ç—•æŠ˜å‡ºå¯¹ç§°çš„ä¸‰>ç»´é«˜å°æˆ–å‡¹é™·ã€‚ä»Žå¯¹ç§°æ€§åˆ†æžä¸éš¾å‘çŽ°ï¼ŒçŽ«ç‘°èŠ±ä¼šæœ‰å››ä¸ªå‘¨å¯¹ç§°çš„ä¸‰ç»´é«˜å°å’Œé…å¥—å‡¹é™·ã€‚æ‰€ä»¥ï¼Œæˆ‘ä»¬å¯ä»¥å…ˆæŠ˜>å‡ºå››åˆ†ä¹‹ä¸€çš„å‡¹é™·å’Œé«˜å°å›¾æ¡ˆï¼Œç„¶åŽä»¥è¿™å››åˆ†ä¹‹ä¸€çš„éƒ¨>åˆ†ä½œä¸ºæ‘¸æ¿ï¼Œå†ä¾æ¬¡æŠ˜å‡ºå…¶ä½™ä¸‰ä¸ªéƒ¨åˆ†çš„é‡å¤å›¾æ¡ˆã€‚å€¼å¾—æ³¨æ„çš„æ˜¯ï¼Œé«˜å°çš„å¸ƒå±€ä¸ä»…è¦è€ƒè™‘é•¿å’Œå®½è¿™ä¸¤ä¸ªå”¯ç‹¬ä¸Šçš„è§„æ•´è¡¬åº¦å’Œå¯¹ç§°åˆ†å¸ƒï¼Œè¿˜éœ€è¦åŒæ—¶ä¿è¯é«˜è¿™ä¸ªç»´åº¦ä¸Š
çš„æ•´é½ã€‚ä¸Žç¬¬ä¸€>é˜¶æ®µçš„æ³¨æ„äº‹é¡¹ç±»ä¼¼ï¼Œè¯·å¤„ç†å¥½ä¸‰ä¸ªç»´åº¦ä¸Šçš„æ‰€æœ‰æŠ˜è§’ï¼Œç¡®ä¿å®ƒä»¬ç¬¦åˆè®¡åˆ’ä¸­æ‰€è¦æ±‚çš„é‚£ç§å¸ƒå±€ï¼Œä»¥å…å‡ºçŽ°ä¸‰ç»´æŠ˜å è¿‡ç¨‹ä¸­çš„è´è¶æ•ˆåº”ï¼›ä¸ºæ­¤ï¼Œæˆ‘ä»¬>å¸¸å¸¸ä¼šåœ¨æŠ˜å ç¬¬ä¸€ä¸ªå››åˆ†ä¹‹ä¸€å›¾æ¡ˆçš„è¿‡ç¨‹ä¸­ï¼Œä¸Žæˆå“çŽ«ç‘°èŠ±è¿›è¡Œåå¤æ¯”è¾ƒï¼Œä»¥ä¾¿åœ¨ç¬¬ä¸€æ—¶>é—´æŽ’é™¤æŽ‰æ‰€æœ‰å¯èƒ½çš„é”™è¯¯ã€‚æœ€åŽä¸€ä¸ªé˜¶æ®µæ˜¯å®ŒæˆèŠ±ç“£ä¿®é¥°ã€‚åœ¨è¿™ä¸ªé˜¶æ®µï¼Œæˆ‘>ä»¬å¾€å¾€å¼ºè°ƒä¸€ä¸ªé‡è¦åè¯ï¼Œå«ç”¨å¿ƒæŠ˜å ã€‚è¿™é‡Œçš„ç”¨å¿ƒå·²ç»ä¸æ˜¯å­—é¢ä¸Šçš„è®¤çœŸè¿™ä¸ªæ„æ€ï¼Œè€Œæ˜¯æŒ‡é€šè¿‡æˆ‘ä»¬å¯¹äºŽå¤§è‡ªç„¶ä¸­çŽ«ç‘°èŠ±å¤–åž‹çš„ç†è§£ï¼Œå€ŸåŠ©è‡ªç„¶çš„æ›²çº¿åŽ»ä¸æ–­ä¿®>æ­£>èŠ±ç“£çš„å½¢çŠ¶ï¼Œä»¥æœŸé€¼è¿‘çŽ°å®žä¸­çš„çŽ«ç‘°èŠ±ç“£å¤–å½¢ã€‚è¯·æ³¨æ„ï¼Œåœ¨è¿™ä¸ªé˜¶æ®µçš„æœ€åŽä¸€æ­¥ï¼Œæˆ‘ä»¬éœ€è¦é€šè¿‡æ‹‰æ‰¯å·²ç»å¼¯æŠ˜çš„å››ä¸ªèŠ±ç“£ï¼Œæ¥è°ƒæ•´çŽ«ç‘°èŠ±ä¸­å¿ƒçš„ç»½æ”¾ç¨‹åº¦ã€‚è¿™ä¸ªè¿‡ç¨‹å¯
èƒ½ä¼šä¼´éšçŽ«ç‘°èŠ±æ•´ä½“ç»“æž„çš„å´©å¡Œï¼Œæ‰€ä»¥ï¼Œä¸€å®šè¦æŽ§åˆ¶å¥½è°ƒæ•´çš„åŠ›é“ï¼Œ>ä»¥å…å‡ºçŽ°ä¸å¯é€†çš„åŽæžœã€‚æœ€ç»ˆï¼Œç»è¿‡ä¸‰ä¸ªé˜¶æ®µçš„æŠ˜å ï¼Œæˆ‘ä»¬ä¼šå¾—åˆ°ä¸€æ”¯æ ©æ ©å¦‚ç”Ÿçš„çŽ«ç‘°èŠ±å† ã€‚å¦‚>æžœæ¡ä»¶å…è®¸ï¼Œæˆ‘ä»¬å¯ä»¥åœ¨ä¸€æ ¹æ‹‰ç›´çš„é“ä¸ä¸Šç¼ ç»•ç»¿è‰²çº¸æ¡ï¼Œå¹¶å°†çŽ«ç‘°èŠ±å† æ’åœ¨é“ä¸çš„ä¸€æ®µã€‚è¿™æ ·ï¼Œæˆ‘ä»¬å°±å¾—åˆ°äº†ä¸€æ”¯æ‰‹å·¥çŽ«ç‘°èŠ±ã€‚æ€»ä¹‹ï¼Œé€šè¿‡>åˆ›å»ºæ …æ ¼æŠ˜ç—•ï¼Œåˆ¶ä½œç«‹>ä½“åŸºåº§ï¼Œä»¥åŠå®ŒæˆèŠ±ç“£ä¿®é¥°ï¼Œæˆ‘ä»¬ä»ŽäºŒç»´çš„çº¸é¢ä¸Šåˆ›ä½œå‡ºäº†ä¸€æ”¯ä¸‰ç»´çš„èŠ±æœµã€‚è¿™ä¸ªè¿‡ç¨‹è™½ç„¶çœ‹ä¼¼ç®€å•ï¼Œä½†å®ƒç¡®å®žæˆ‘ä»¬äººç±»å€ŸåŠ©æƒ³è±¡åŠ›å’Œå¸¸è§ç´ æè€Œåˆ›ä½œå‡ºçš„è‰ºæœ¯å“ã€‚é—®
: è¯·åŸºäºŽä»¥ä¸Šæè¿°ï¼Œåˆ†æžå“ªäº›æ­¥éª¤åšé”™äº†å¾ˆå¤§å¯>èƒ½ä¼šå¯¼è‡´æœ€ç»ˆæŠ˜å å¤±è´¥ï¼Ÿç­”: """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--precision', type=str, default='bf16',
                        help='Main model Precision')
    parser.add_argument('--n_predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--max-draft', type=int, default=8,
                        help='Max draft')


    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    # Load model in optimized bf16 here.
    # Set `speculative=True`` to enable speculative decoding,
    # it only works when load_in_low_bit="fp16" on Intel GPU or load_in_low_bit="bf16" on latest Intel Xeon CPU
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 torch_dtype=torch.bfloat16,
                                                 load_in_low_bit="bf16",
                                                 speculative=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with torch.inference_mode():
        prompt = BAICHUAN_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                do_sample=False)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                do_sample=False)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        end = time.perf_counter()

        print(output_str)
        print(f"Tokens generated {model.n_token_generated}")
        print(f"E2E Generation time {(end - st):.4f}s")
        print(f"First token latency {model.first_token_time:.4f}s")
