import os
from tqdm import tqdm
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
from accelerate import init_empty_weights,load_checkpoint_and_dispatch
from evaluators.evaluator import Evaluator
from time import sleep
import re


class Moss_Evaluator(Evaluator):
    def __init__(self,choices,k,model_name):
        super(Moss_Evaluator,self).__init__(choices,model_name,k)
        model_path="fnlp/moss-moon-003-sft"
        if not os.path.exists(model_path):
            model_path=snapshot_download(model_path)
        self.config=AutoConfig.from_pretrained("fnlp/moss-moon-003-sft",trust_remote_code=True)
        self.tokenizer=AutoTokenizer.from_pretrained("fnlp/moss-moon-003-sft",trust_remote_code=True)
        self.tokenizer.padding_side="left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with init_empty_weights():
            self.model=AutoModelForCausalLM.from_config(self.config,torch_dtype=torch.float16,trust_remote_code=True)
        self.model.tie_weights()
        self.model=load_checkpoint_and_dispatch(self.model,model_path,device_map="auto",no_split_module_classes=["MossBlock"],
                                           dtype=torch.float16)

    def format_example(self,line,include_answer=True,cot=False):
        example=line['question']
        for choice in self.choices:
            example+=f'\n{choice}. {line[f"{choice}"]}'

        example+='\n答案：'
        if include_answer:
            if cot:
                ans=line["answer"]
                content="让我们一步一步思考，\n"+line["explanation"]+f"\n所以答案是{ans}。"
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":content}
                ]
            else:
                return [
                    {"role":"user","content":example},
                    {"role":"assistant","content":line["answer"]}
                ]
        else:
            return [
                {"role":"user","content":example},
            ]

    def generate_few_shot_prompt(self,subject,dev_df,cot=False):
        prompt=f"你是一个中文人工智能助手，以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n"
        k=self.k
        if self.k==-1:
            k=dev_df.shape[0]
        for i in range(k):
            tmp=self.format_example(dev_df.iloc[i,:],include_answer=True,cot=cot)
            if i==0:
                tmp[0]["content"]=f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n"+tmp[0]["content"]
            user=tmp[0]['content']
            moss=tmp[1]['content']
            prompt+=f"<|Human|>: {user} <eoh>\n<|MOSS|>: {moss} <eom>\n"
        return prompt

    def eval_subject(self,subject_name,test_df,dev_df=None,few_shot=False,save_result_dir=None,cot=False):
        correct_num=0
        if save_result_dir:
            result=[]
            score=[]
        if few_shot:
            few_shot_prompt=self.generate_few_shot_prompt(subject_name,dev_df,cot=cot)
        else:
            few_shot_prompt=f"你是一个中文人工智能助手，以下是中国关于{subject_name}考试的单项选择题，请选出其中的正确答案。\n"
        answers=list(test_df['answer'])
        message_list=[]
        tar_list=[]
        for row_index,row in tqdm(test_df.iterrows(),total=len(test_df)):
            question=self.format_example(row,include_answer=False)
            full_prompt=few_shot_prompt+"<|Human|>: "+question[0]['content']+" <eoh>\n<|MOSS|>:"
            message_list.append(full_prompt)
            tar_list.append(answers[row_index])
            if len(message_list)%1==0 or row_index==len(test_df)-1:
                inputs=self.tokenizer(message_list,return_tensors="pt",padding=True)
                for k in inputs:
                    inputs[k]=inputs[k].cuda()
                max_tok=2040-inputs.input_ids.shape[1]
                outputs=self.model.generate(**inputs,do_sample=True,temperature=0.2,top_p=0.8,repetition_penalty=1.02,
                                            max_new_tokens=max_tok)
                input_len=torch.max(torch.sum(inputs.attention_mask,axis=1))
                response_list=[
                    self.tokenizer.decode(outputs[i][input_len:],skip_special_tokens=True)
                    for i in range(outputs.shape[0])
                ]
                for i,response_str in enumerate(response_list):
                    #print(response_str)
                    if cot:
                        ans_list=re.findall(r"答案是(.+?)。",response_str)
                        if len(ans_list)==0:
                            ans_list=re.findall(r"答案为(.+?)。",response_str)
                        if len(ans_list)==0:
                            ans_list=re.findall(r"选项(.+?)是正确的。",response_str)

                        if len(ans_list)==0:
                            correct=0
                        else:
                            if self.exact_match(ans_list[-1],tar_list[i]):
                                correct_num+=1
                                correct=1
                            else:
                                correct=0
                    else:
                        response_str=response_str.strip()
                        if few_shot:
                            if self.exact_match(response_str,tar_list[i]):
                                correct_num+=1
                                correct=1
                            else:
                                correct=0
                        else:
                            if response_str[0]==tar_list[i]:
                                correct_num+=1
                                correct=1
                            else:
                                correct=0

                    if save_result_dir:
                        result.append(response_str)
                        score.append(correct)
                message_list=[]
                tar_list=[]

        correct_ratio=100*correct_num/len(answers)

        if save_result_dir:
            test_df['model_output']=result
            test_df["correctness"]=score
            test_df.to_csv(os.path.join(save_result_dir,f'{subject_name}_val.csv'),encoding="utf-8",index=False)
        return correct_ratio
