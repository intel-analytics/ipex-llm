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
# refer to https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/blob/main/scripts/ceval/llama_evaluator.py

import re
import random
from tqdm import tqdm
import numpy as np
import torch
from transformers import LlamaTokenizer, GenerationConfig

from ipex_llm.transformers import AutoModelForCausalLM
from evaluators.evaluator import Evaluator


DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""


class LlamaEvaluator(Evaluator):
    def __init__(self, choices, model_path="meta-llama/Llama-2-7b-chat-hf", device="xpu", qtype="sym_int4"):
        super(LlamaEvaluator, self).__init__(choices, model_path, device, qtype)
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_low_bit=self.qtype,
            optimize_model=True,
            use_cache=True,
            trust_remote_code=True
        ).eval().to(self.device)
        self.generation_config = GenerationConfig(
            temperature=0.2,
            top_k=40,
            top_p=0.9,
            do_sample=True,
            num_beams=1,
            repetition_penalty=1.1,
            max_new_tokens=20
        )
        self.sA_id = self.tokenizer.encode("A", add_special_tokens=False)[0]
        self.sB_id = self.tokenizer.encode("B", add_special_tokens=False)[0]
        self.sC_id = self.tokenizer.encode("C", add_special_tokens=False)[0]
        self.sD_id = self.tokenizer.encode("D", add_special_tokens=False)[0]
        self.A_id = self.tokenizer.encode("：A")[-1]
        self.B_id = self.tokenizer.encode("：B")[-1]
        self.C_id = self.tokenizer.encode("：C")[-1]
        self.D_id = self.tokenizer.encode("：D")[-1]


    @torch.no_grad()
    def eval_subject(self, subject_name,
            test_df,
            eval_type="validation",
            dev_df=None,
            few_shot=False,
            cot=False,
            with_prompt=False,
            constrained_decoding=False):
        all_answers = {}
        if constrained_decoding is True:
            self.generation_config.output_scores = True
            self.generation_config.return_dict_in_generate = True
            self.generation_config.max_new_tokens = 1
            self.generation_config.top_p = 1.0
            self.generation_config.top_k = 0

        correct_num = 0
        if few_shot:
            if with_prompt:
                history = self.generate_alpaca2_few_shot_prompt(subject_name, dev_df, cot=cot)
            else:
                history = self.generate_llama2_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = ''
        answers = ['NA'] * len(test_df) if (eval_type=="test") is True else list(test_df['answer'])
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot,with_prompt=with_prompt)
            instruction = question
            if with_prompt:
                prompt_template = (
                                        "[INST] <<SYS>>\n"
                                        "{system_prompt}\n"
                                        "<</SYS>>\n\n"
                                        "{instruction} [/INST]"
                                    )

                instruction = prompt_template.format_map({'instruction': instruction,'system_prompt':DEFAULT_SYSTEM_PROMPT})
            instruction = history + instruction
            inputs = self.tokenizer(instruction, return_tensors="pt")
            generation_output = self.model.generate(
                    input_ids = inputs["input_ids"].to(self.device),
                    attention_mask = inputs['attention_mask'].to(self.device),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    generation_config = self.generation_config
                )

            _ , length = inputs.input_ids.shape
            if constrained_decoding is True:
                logits = generation_output.scores[0][0]

                logits = logits.float().cpu().detach()
                choices1_logits = logits[[self.sA_id,self.sB_id,self.sC_id,self.sD_id]]
                choices2_logits = logits[[self.A_id,self.B_id,self.C_id,self.D_id]]
                choicesAll_logits = (choices1_logits + choices2_logits).numpy()
                assert not (np.any(np.isinf(choicesAll_logits)) or np.any(np.isnan(choicesAll_logits)))
                ans = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(choicesAll_logits)]
                response = self.tokenizer.decode([logits.argmax(-1).item()])
            else:
                response = self.tokenizer.decode(generation_output[0, length:], skip_special_tokens=True)
                ans, _ = self.extract_answer(response, row)
            if ans == answers[row_index]:
                correct_num += 1

            all_answers[str(row_index)] = ans

        correct_ratio = 100*correct_num/len(answers)

        return correct_ratio, all_answers


    def format_example(self, line, include_answer=True, cot=False, with_prompt=False):
        example = line['question']
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        if include_answer:
            if cot:
                example += "\n答案：让我们一步一步思考，\n" + \
                    line["explanation"] + f"\n所以答案是{line['answer']}。\n\n"
            else:
                example += '\n答案：' + line["answer"] + '\n\n'
        else:
            if with_prompt is False:
                if cot:
                    example += "\n答案：让我们一步一步思考，\n1."
                else:
                    example += '\n答案：'
            else:
                if cot:
                    example += "\n答案是什么？让我们一步一步思考，\n1."
                else:
                    example += '\n答案：'
        return example


    def generate_llama2_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            prompt += self.format_example(
                dev_df.iloc[i, :],
                include_answer=True,
                cot=cot
            )
        return prompt


    def generate_alpaca2_few_shot_prompt(self, subject, dev_df, cot=False):
        prompt = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"
        prompt_template = (
            "[INST] <<SYS>>\n"
            "{system_prompt}\n"
            "<</SYS>>\n\n"
            "{instruction} [/INST]好的，我会结合{subject}相关知识回答"
        )

        prompt = prompt_template.format_map({'instruction':prompt,'system_prompt':DEFAULT_SYSTEM_PROMPT,'subject':subject})
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        for i in range(k):
            line = dev_df.iloc[i, :]
            q=line['question']
            for choice in self.choices:
                q += f'\n{choice}. {line[f"{choice}"]}'

            a = line['answer']
            prompt += "[INST] "+q+"\n答案：[/INST]"+a+"\n"
        return prompt


    def extract_answer(self, response, row):
        m = re.findall(r'所以答案是(.+?)。', response, re.M)
        if len(m) > 0 and m[-1] in self.choices:
            return m[-1], True
        answer_patterns = [
            r'([ABCD])是正确的',
            r'选项([ABCD])正确',
            r'答案为([ABCD])',
            r'答案是([ABCD])',
            r'答案([ABCD])',
            r'选择([ABCD])',
            r'答案：([ABCD])',
            r'选择答案([ABCD])'
        ]
        # RE extraction
        for answer_pattern in answer_patterns:
            m = re.search(answer_pattern, response, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', response, re.M)
        if len(m) >= 1:
            answer = m[0]
            return answer, False
        # only containing one choice-context
        choices_dict = {}
        pattern = ""
        for c in self.choices:
            choices_dict[str(row[f'{c}'])] = c
            pattern += re.escape(str(row[f'{c}']))+"|"
        pattern = pattern[:-1]
        m = re.findall(pattern, response, re.M)
        print("w/ escape:",repr(pattern),response,(len(m)>=1))
        if len(m) >= 1:
            answer = choices_dict[m[0]]
            return answer, False
        return  random.choice('ABCD'), False
