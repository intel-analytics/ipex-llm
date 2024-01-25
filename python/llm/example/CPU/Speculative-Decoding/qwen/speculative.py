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
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
QWEN_PROMPT_FORMAT = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

long_input = """患者男，年龄29岁，血型O，因思维迟钝，易激怒，因发热伴牙龈出血14天，乏力、头晕5天就诊我院急诊科。快速完善检查，血常规显示患者三系血>细胞重度减低，凝血功能检查提示APTT明显延长，纤维蛋白原降低，血液科会诊后发现患者高热、牙龈持续出血，胸骨压痛阳性.于3903年3月7日入院治疗，出现头痛、头晕、伴发热（最高体温42℃）症状，曾到其他医院就医。8日症状有所好转，9日仍有头痛、呕吐，四肢乏力伴发热。10日凌晨到本院就诊。患者5d前出现突发性思维迟钝，脾气暴躁，略有不顺心就出现攻击行为，在院外未行任何诊治。既往身体健康，平素性格内向。体格检查无>异常。血常规白细胞中单核细胞百分比升高。D-二聚体定量1412μg/L，骨髓穿刺示增生极度活跃，异常早幼粒细胞占94%.外周血涂片见大量早幼粒细>胞，并可在胞浆见到柴捆样细胞.以下是血常规详细信息：1.病人红细胞计数结果：3.2 x10^12/L. 附正常参考范围：新生儿:（6.0～7.0）×10^12/L>；婴儿：（5.2～7.0）×10^12/L; 儿童：（4.2～5.2）×10^12/L; 成人男：（4.0～5.5）×10^12/L; 成人女：（3.5～5.0）×10^12/L. 临床意义：生>理性红细胞和血红蛋白增多的原因：精神因素（冲动、兴奋、恐惧、冷水浴刺激等导致肾上腺素分泌增多的因素）、红细胞代偿性增生（长期低气压>、缺氧刺激，多次献血）；生理性红细胞和血红蛋白减少的原因：造血原料相对不足，多见于妊娠、6个月～2岁婴幼儿、某些老年性造血功能减退；>病理性增多：多见于频繁呕吐、出汗过多、大面积烧伤、血液浓缩，慢性肺心病、肺气肿、高原病、肿瘤以及真性红细胞增多症等；病理性减少：多>见于白血病等血液系统疾病；急性大出血、严重的组织损伤及血细胞的破坏等；合成障碍，见于缺铁、维生素B12缺乏等。2. 病人血红蛋白测量结果>：108g/L. 附血红蛋白正常参考范围：男性120～160g/L；女性110～150g/L；新生儿170～200g/L；临床意义：临床意义与红细胞计数相仿，但能更好地反映贫血程度，极重度贫血（Hb<30g/L）、重度贫血（31～60g/L）、中度贫血（61～90g/L）、男性轻度贫血（90~120g/L）、女性轻度贫血（90~110g/L）。3. 病人白细胞计数结果：13.6 x 10^9/L; 附白细胞计数正常参考范围：成人（4.0～10.0）×10^9/L；新生儿（11.0～12.0）×10^9/L。临>床意义：1）生理性白细胞计数增高见于剧烈运动、妊娠、新生儿；2）病理性白细胞增高见于急性化脓性感染、尿毒症、白血病、组织损伤、急性出>血等；3）病理性白细胞减少见于再生障碍性贫血、某些传染病、肝硬化、脾功能亢进、放疗化疗等。4. 病人白细胞分类技术结果：中性粒细胞（N）50%、嗜酸性粒细胞（E）3.8%、嗜碱性粒细胞（B）0.2%、淋巴细胞（L）45%、单核细胞（M）1%。附白细胞分类计数正常参考范围：中性粒细胞（N）50%～70%、嗜酸性粒细胞（E）1%～5%问：请基于以上信息做出判断，该患者是否有罹患急性白血病的风险？请结合上述内容给出判断的详细解释，并简要总结潜在的早期征兆、预防方法、常用的治疗手段。答：" """

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Qwen model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen-7B-Chat",
                        help='The huggingface repo id for the Qwen (e.g. `Qwen/Qwen-7B-Chat` and `Qwen/Qwen-14B-Chat`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--th_stop_draft', type=float, default=0.6,
                        help='draft stop probility')

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
    model = model.to('cpu')

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with torch.inference_mode():
        prompt = QWEN_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)
        actual_in_len = input_ids.shape[1]
        print("actual input_ids length:" + str(actual_in_len))

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                th_stop_draft=args.th_stop_draft,
                                do_sample=False)
        output_str = tokenizer.decode(output[0])

        # speculative decoding
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                th_stop_draft=args.th_stop_draft,
                                do_sample=False)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        end = time.perf_counter()

        print(output_str)
        print(f"Tokens generated {model.n_token_generated}")
        print(f"E2E Generation time {(end - st):.4f}s")
        print(f"First token latency {model.first_token_time:.4f}s")
