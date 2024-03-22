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
# refer to https://github.com/QwenLM/Qwen/blob/main/eval/evaluate_chat_ceval.py

import re
from tqdm import tqdm
import torch
from thefuzz import process
from transformers import AutoTokenizer
from transformers.generation import GenerationConfig

from ipex_llm.transformers import AutoModelForCausalLM
from evaluators.evaluator import Evaluator


class QwenEvaluator(Evaluator):
    def __init__(self, choices, model_path="Qwen/Qwen-7B-Chat", device="xpu", qtype="sym_int4"):
        super(QwenEvaluator, self).__init__(choices, model_path, device, qtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
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
        self.model.generation_config = GenerationConfig.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model.generation_config.do_sample = False  # use greedy decoding
        self.model.generation_config.repetition_penalty = 1.0  # disable repetition penalty


    def process_before_extraction(self, gen, question, choice_dict):

        question_split = question.rstrip("。").split("。")[-1].split("_")

        if len(question_split[0].strip()) > 4:
            gen = gen.replace(question_split[0], "答案是")
        if len(question_split[-1].strip()) > 4:
            gen = gen.replace(question_split[-1], "")

        for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
            gen = gen.replace(val.rstrip("。"), key)
        return gen


    def count_substr(self, gen, pattern):
        return len(re.findall(pattern, gen))


    def extract_choice(self, gen, prompt, choice_list):
        res = re.search(
            r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCD]{0,10}?(?:不|非)[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?)(A|B|C|D)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|$|：|:|\)|）)",
            gen,
        )

        if res is None:
            res = re.search(
                r"(A|B|C|D)(?:选?项)?(?![^ABCD]{0,4}?(?:不|非)[^ABCD]{0,4}?(?:正确|对[的，。：]|符合))[^ABCD]{0,4}?(?:正确|对[的，。：]|符合)",
                gen,
            )

        if res is None:
            res = re.search(r"^[\(（]?(A|B|C|D)(?:。|\)|）|\.|，|,|．|：|:|$)", gen)

        if res is None:
            res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)

        if res is None:
            return self.choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
        return res.group(1)


    def format_example(self, line):
        example = line["question"] + "\n\n"
        for choice in self.choices:
            example += f'{choice}. {line[f"{choice}"]}\n'
        return example


    def extract_answer(self, response, row):
        prompt = row["question"]
        gen = self.process_before_extraction(
            response, prompt, {choice: row[choice] for choice in self.choices}
        )
        if not isinstance(prompt, str):
            prompt = prompt[0]
        pred = self.extract_choice(gen, prompt, [row[choice] for choice in self.choices])
        return pred


    @torch.no_grad()
    def eval_subject(
        self,
        subject_name,
        test_df,
        eval_type="validation" # "test","validation"
    ):
        if eval_type == "validation":
            responses = []
            result = []
            score = []
            for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row)

                response, _ = self.model.chat(
                    self.tokenizer,
                    question,
                    history=None,
                )
                pred = self.extract_answer(response, row)
                if "answer" in row:
                    correct = 1 if pred == row["answer"] else 0
                    score.append(correct)
                responses.append(response)
                result.append(pred)

            if score:
                correct_ratio = 100 * sum(score) / len(score)

            else:
                correct_ratio = 0

            return correct_ratio, None
        elif eval_type == "test":
            answers = {}
            for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row)
                response, _ = self.model.chat(
                    self.tokenizer,
                    question,
                    history=None,
                )
                pred = self.extract_answer(response, row)
                answers[str(i)] = pred
            return None, answers
