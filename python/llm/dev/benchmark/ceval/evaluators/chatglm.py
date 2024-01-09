import os
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from evaluators.evaluator import Evaluator
from bigdl.llm.transformers import AutoModelForCausalLM

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores

class ChatGLMEvaluator(Evaluator):
    def __init__(self, choices, k, model_name, device):
        super(ChatGLMEvaluator, self).__init__(choices, model_name, k)
        # try adding 'mirror="tuna"' and 'resume_download=True' if facing the 'read timed out' problem
        # or directly clone the model
        # self.tokenizer = AutoTokenizer.from_pretrained("/disk7/zcg/chatglm2-6b", trust_remote_code=True, mirror="tuna")
        # self.model = AutoModel.from_pretrained("/disk7/zcg/chatglm2-6b", trust_remote_code=True, mirror="tuna", resume_download=True, torch_dtype=torch.bfloat16).to(device, dtype=float)
        self.tokenizer = AutoTokenizer.from_pretrained("/mnt/disk1/models/Qwen-7B-Chat/", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("/mnt/disk1/models/Qwen-7B-Chat/", trust_remote_code=True, load_in_low_bit="sym_int4").to(device)

    def eval_subject(self, subject_name, test_df, dev_df=None, few_shot=True, cot=False, save_result_dir=None):
        all_answers = {}
        if few_shot:
            history = self.generate_few_shot_prompt(subject_name, dev_df, cot=cot)
        else:
            history = []
        for row_index, row in tqdm(test_df.iterrows(), total=len(test_df)):
            question = self.format_example(row, include_answer=False, cot=cot)
            if few_shot:
                response, _ = self.model.chat(self.tokenizer, question, do_sample=False, history=history)
                response = response.strip()
                # For ChatGLM, we use answer extraction in answer-only mode too.
                ans, direct_extract = self.extract_cot_answer(row, response)
            else:   # zero-shot by extracting answer from distribution
                ans = self.generate_dist(self.model, self.tokenizer, question, do_sample=False, max_length=2048, history=history)
            all_answers[str(row_index)] = ans
        
        return all_answers
    
    def generate_few_shot_prompt(self, subject, dev_df, cot=False):
        message = []
        k = self.k
        if self.k == -1:
            k = dev_df.shape[0]
        message.append(self.format_example(dev_df.iloc[0, :], cot=cot, add_prompt=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"))
        for i in range(1, k):
            message.append(self.format_example(dev_df.iloc[i, :], cot=cot))
        return message
        
    def format_example(self, line, include_answer=True, cot=False, add_prompt=''):
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
    
    def generate_dist(self, model, tokenizer, query, history, num_beams=1, max_length=2048,
                      do_sample=False, top_p=0.7, temperature=0.95, logits_processor=None, **kwargs):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"num_beams": num_beams, "do_sample": do_sample, "top_p": top_p, "max_length": 2048,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n问：{}\n答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
        inputs = tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(model.device)
        outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, **gen_kwargs)
        
        score = outputs.scores[0][0].tolist()
        choice_score = [score[167], score[333], score[251], score[416]]
        ranked_index = [index for index, value in sorted(list(enumerate(choice_score)), key=lambda x:x[1], reverse=True)]
        return self.choices[ranked_index[0]]
