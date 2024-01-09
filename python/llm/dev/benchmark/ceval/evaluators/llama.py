import os
import re
from typing import List

import numpy as np
import torch
from evaluators.evaluator import Evaluator
from tqdm import tqdm
import transformers
from bigdl.llm.transformers import AutoModelForCausalLM


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


class LLaMA_Evaluator(Evaluator):

    def __init__(self, choices, device, k=-1) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("/mnt/disk1/models/Qwen-7B-Chat/", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("/mnt/disk1/models/Qwen-7B-Chat/", trust_remote_code=True, load_in_low_bit="sym_int4").to(device)
        self.choices = choices
        self.model_name = "LLaMA"
        self.k = k
        self.device = device
        self.patterns = [
            "答案是?\s?([ABCD])",
            "答案是?\s?：([ABCD])",
            "答案是?\s?:([ABCD])",
            "答案应该?是\s?([ABCD])",
            "答案应该?选\s?([ABCD])",
            "答案为\s?([ABCD])",
            "选择\s?([ABCD])",
            "只有选?项?\s?([ABCD])\s?是?对",
            "只有选?项?\s?([ABCD])\s?是?错",
            "只有选?项?\s?([ABCD])\s?不?正确",
            "只有选?项?\s?([ABCD])\s?错误",
            "说法不?对选?项?的?是\s?([ABCD])",
            "说法不?正确选?项?的?是\s?([ABCD])",
            "说法错误选?项?的?是\s?([ABCD])",
            "([ABCD])\s?是正确的",
            "([ABCD])\s?是正确答案",
            "选项\s?([ABCD])\s?正确",
            "所以答\s?([ABCD])",
            "1.\s?([ABCD])[.。$]?$",
            "所以\s?([ABCD][.。$]?$)",
            "所有\s?([ABCD][.。$]?$)",
            "[\s，：:,]([ABCD])[。，,\.]?$",
            "[\s，,：:][故即]([ABCD])[。\.]?$",
            "[\s，,：:]因此([ABCD])[。\.]?$",
            "[是为。]\s?([ABCD])[。\.]?$",
            "因此\s?([ABCD])[。\.]?$",
            "显然\s?([ABCD])[。\.]?$",
            "1.\s?(.*?)$",
            "答案是\s?(\S+)(?:。|$)",
            "答案应该是\s?(\S+)(?:。|$)",
            "答案为\s?(\S+)(?:。|$)",
        ]

    def format_example(self, line, include_answer=True, cot=False):
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
            if cot:
                example += "\n答案：让我们一步一步思考，\n1."
            else:
                example += '\n答案：'
        return example

    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
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

    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_logits: bool = False
    ) -> List[str]:
        params = self.model.params
        prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False).to(self.device)
        prompt_size = len(prompt_tokens)
        total_len = min(params.max_seq_len, max_gen_len + prompt_size)

        tokens = torch.full(
            (1, total_len), self.tokenizer.pad_id).long().to(self.device)
        tokens[0, : prompt_size] = torch.tensor(prompt_tokens).long().to(self.device)
        input_text_mask = tokens != self.tokenizer.pad_id
        prev_pos = 0
        if return_logits:
            return self.model.forward(tokens[:, :prompt_size], 0)
        for cur_pos in range(prompt_size, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos

        decoded = []
        for _, t in enumerate(tokens.tolist()):
            t = t[: prompt_size + max_gen_len]
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))
        return decoded
    
    def extract_model_answer(self,text, a,b,c,d):
        option_str=re.escape('A. '+a+'\nB. '+b+'\nC. '+c+'\nD. '+d)
        match = re.search(rf'{option_str}([\s\S]*)$', text)
        if match:
            return match.group(1)
        else:
            return None
    
    def extract_answer_option(self,text):
        match = re.findall(r'(让我们一步一步思考[\s\S]+?)(?:(?=让我们一步一步思考)|$)', text)
        text=match[0]
        regexes = [re.compile(pattern) for pattern in self.patterns]
        for regex in regexes:
            match = regex.search(text)
            if match:
                return match.group(1)
        return None
    
    def answer_str(self,answer,a,b,c,d):
        if answer=='D':
            return d
        elif answer=='C':
            return c
        elif answer=='B':
            return b
        else:
            return a

    def extract_answer(self, row, output):
        pred = {"A":0, "B":1, "C":2, "D":3}
        correct_answer_str=self.answer_str(row['answer'], row['A'],row['B'],row['C'],row['D'])
        generate_answer=self.extract_model_answer(str(output), row['A'],row['B'],row['C'],row['D'])
        model_answer=self.extract_answer_option(generate_answer)
        if row['answer']==model_answer or correct_answer_str==model_answer:
            return pred[model_answer] if model_answer in pred else model_answer, 1
        else:
            return pred[model_answer] if model_answer in pred else model_answer, 0

    def eval_subject(
        self,
        subject_name,
        test_df,
        dev_df=None,
        few_shot=False,
        save_result_dir=None,
        cot=False,
        **kwargs
    ):
        all_answers = {}
        few_shot_prompt = self.generate_few_shot_prompt(
            subject_name, dev_df, cot=cot) if few_shot else "" 
        for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            full_prompt = few_shot_prompt + question
            output = self.generate(
                full_prompt,
                max_gen_len=kwargs.get("max_gen_len", 512),
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95),
                return_logits=not cot
            )
            if cot:
                assert isinstance(output[0], str)
                output = output[0]
                pred, correct = self.extract_answer(row, output)
            else:
                assert output.shape[0] == 1
                logits = output.flatten()
                probs = (
                    torch.nn.functional.softmax(
                        torch.tensor(
                            [
                                logits[self.tokenizer.encode(
                                    "A", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "B", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "C", bos=False, eos=False)[0]],
                                logits[self.tokenizer.encode(
                                    "D", bos=False, eos=False)[0]],
                            ]
                        ),
                        dim=0,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(probs)]
            all_answers[i] = pred
        
        return all_answers
