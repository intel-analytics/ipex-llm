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
# refer to https://github.com/THUDM/ChatGLM2-6B/blob/main/evaluation/evaluate_ceval.py

import re
import torch
from tqdm import tqdm
from thefuzz import process
from transformers import AutoTokenizer

from evaluators.evaluator import Evaluator
from ipex_llm.transformers import AutoModel
from transformers.generation.utils import LogitsProcessorList
from transformers.generation.logits_process import LogitsProcessor


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLMEvaluator(Evaluator):
    def __init__(self, choices, model_path="THUDM/chatglm-6b", device="xpu", qtype="sym_int4"):
        super(ChatGLMEvaluator, self).__init__(choices, model_path, device, qtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self.model_path,
            load_in_low_bit=self.qtype,
            optimize_model=True,
            use_cache=True,
            trust_remote_code=True
        ).eval().to(self.device)
  
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        message = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        message.append(self.format_example(dev_df.iloc[0, :], cot=cot, add_prompt=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"))
        for i in range(1, k):
            message.append(self.format_example(dev_df.iloc[i, :], cot=cot))
        return message
        
    def format_example(self, line, include_answer=False, cot=False, add_prompt=''):
        example = add_prompt + line['question']
        # print(example)
        for choice in self.choices:
            example += f'\n{choice}. {line[f"{choice}"]}'
        example += '\n答案：'
        if include_answer:
            if cot:
                ans = "让我们一步一步思考，\n" + line["explanation"] + f"\n所以答案是{line['answer']}。"
            else:
                ans = line["answer"]
            m = (example, ans)
            return m
        return example
    
    def extract_cot_answer(self, line, gen_ans):
        m = re.findall(r'所以答案是(.+?)。', gen_ans, re.M)
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
            m = re.search(answer_pattern, gen_ans, re.M)
            if m:
                answer = m.group(1)
                return answer, False
        # only containing one choice-character
        m = re.findall(r'[ABCD]', gen_ans, re.M)
        if len(m) == 1:
            answer = m[0]
            return answer, False
        answer_word_counter = 0
        # only containing one choice-context
        for c in self.choices:
            if str(line[f'{c}']) in gen_ans:
                answer = c
                answer_word_counter += 1
        if answer_word_counter == 1:
            return answer, False
        return '-', False
    
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

    def process_before_extraction(self, gen, question, choice_dict):

        question_split = question.rstrip("。").split("。")[-1].split("_")

        if len(question_split[0].strip()) > 4:
            gen = gen.replace(question_split[0], "答案是")
        if len(question_split[-1].strip()) > 4:
            gen = gen.replace(question_split[-1], "")

        for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
            gen = gen.replace(val.rstrip("。"), key)
        return gen

    def extract_answer(self, response, row):
        prompt = row["question"]
        gen = self.process_before_extraction(
            response, prompt, {choice: row[choice] for choice in self.choices}
        )
        if not isinstance(prompt, str):
            prompt = prompt[0]
        pred = self.extract_choice(gen, prompt, [row[choice] for choice in self.choices])
        return pred

    def build_prompt(self, text):
        return "[Round {}]\n\n问：{}\n\n答：".format(1, text)

    def generate_dist(self, model, tokenizer, query, history, max_length=2048,
                      do_sample=False, logits_processor=None):
        
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())

        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)

        # first round prompt
        inputs = tokenizer([prompt], padding=True, return_tensors="pt",
                           truncation=True, max_length=max_length).to(model.device)
        
        # first round generation
        outputs = model.generate(**inputs, do_sample=do_sample, max_new_tokens=512)

        # organize intermediate_outputs
        intermediate_outputs = []
        for idx in range(len(outputs)):
            output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
            response = tokenizer.decode(output)
            intermediate_outputs.append(response)

        # prepare second round prompt
        extraction_prompt = '综上所述，ABCD中正确的选项是：'
        answer_texts = [query + intermediate + "\n" + extraction_prompt for intermediate in intermediate_outputs]
        input_tokens = [self.build_prompt(answer_text) for answer_text in answer_texts]
        inputs = tokenizer(input_tokens, padding=True, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        # second round generation
        outputs = model(**inputs, return_last_logit=True)

        logits = outputs.logits[:, -1]
        choice_tokens = [tokenizer.encode(choice, add_special_tokens=False)[0] for choice in self.choices]
        logits = logits[:, choice_tokens]
        preds = logits.argmax(dim=-1)

        return self.choices[preds]

    @torch.no_grad()
    def eval_subject(
        self,
        subject_name,
        test_df,
        eval_type="validation", # "test","validation",
        dev_df=None,
        few_shot=False,
        cot=True,
    ):
        if eval_type == "validation":
            correct_num = 0

            if few_shot:
                history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
            else:
                history = []

            answers = list(test_df['answer'])

            for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row, include_answer=False, cot=cot)

                if few_shot:
                    response, _ = self.model.chat(self.tokenizer, question, do_sample=False, history=history)
                    response = response.strip()
                    # For ChatGLM, we use answer extraction in answer-only mode too.
                    ans, direct_extract = self.extract_cot_answer(row, response)
                else:   # zero-shot by extracting answer from distribution
                    ans = self.generate_dist(self.model, self.tokenizer, question, do_sample=False, max_length=2048, history=history)

                if ans == answers[row_index]:
                    correct_num += 1

            correct_ratio = 100*correct_num/len(answers)

            return correct_ratio, None
        elif eval_type == "test":
            answers = {}
            for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
                question = self.format_example(row, include_answer=False, cot=cot)
                answers[str(i)] = self.generate_dist(self.model, self.tokenizer, question, do_sample=False, max_length=2048, history=[])

            return None, answers