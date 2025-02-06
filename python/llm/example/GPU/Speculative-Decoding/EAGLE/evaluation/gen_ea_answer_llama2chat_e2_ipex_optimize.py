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
# This file is adapted from
# https://github.com/SafeAILab/EAGLE/blob/main/eagle/evaluation/gen_ea_answer_llama2chat.py
#
# Copyright 2024 SafeAI Lab (SAIL).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import json
import os
import time

import torch

import shortuuid
from fastchat.llm_judge.common import load_questions
from fastchat.model import get_conversation_template
from tqdm import tqdm

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor, evaluate_posterior
from eagle.model.kv_cache import initialize_past_key_values
from eagle.model.choices import *
from eagle.modeling_eagle import forward_with_tree_mask
from ipex_llm import optimize_model
from ipex_llm.transformers import AutoModelForCausalLM

class Timer:
    def __init__(self,name):
        self.name = name

    def __enter__(self):
        torch.xpu.synchronize()
        self.start = time.perf_counter()


    def __exit__(self, exc_type, exc_value, traceback):
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - self.start
        print(f'{self.name} took {elapsed} seconds')


def initialize_tree(input_ids, model, logits_processor):
    # outputs, orig, hidden_states = model(
    #     input_ids, past_key_values=past_key_values, output_orig=True
    # )
    hidden_states, past_key_values = forward_with_tree_mask(model.base_model.model, input_ids=input_ids)
    orig = model.base_model.lm_head(hidden_states)

    if logits_processor is not None:
        logits = orig[:, -1]
        logits = logits_processor(None, logits)
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        token = torch.multinomial(probabilities, 1)
    else:
        token = torch.argmax(orig[:, -1])
        token = token[None, None]
    input_ids = torch.cat((input_ids, token.to(input_ids.device)), dim=1)
    # Clone the output hidden states

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(hidden_states, input_ids, model.base_model.lm_head,logits_processor)
    return draft_tokens, retrieve_indices,tree_mask,tree_position_ids, orig, hidden_states, token, past_key_values

def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
        attention_mask=None,
        tree_mask=None,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    hidden_states, past_key_values = forward_with_tree_mask(model.base_model.model, input_ids=tree_candidates, past_key_values=past_key_values,
                           position_ids=position_ids, tree_mask=tree_mask, attention_mask=attention_mask)

    tree_logits = model.base_model.lm_head(hidden_states)

    logits = tree_logits[0, retrieve_indices]

    return logits, hidden_states, past_key_values


def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        new_token,
        past_key_values,
        # current_length_data,
        model,
        hidden_state_new,
        sample_p
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )

    new_kv = ()

    for past_key_values_data in past_key_values:
        layer_kv = ()
        for korv in past_key_values_data:
            tgt = korv[:, :, select_indices, :]
            dst = korv[:, :, prev_input_len: prev_input_len + tgt.shape[-2], :]
            dst.copy_(tgt, non_blocking=True)
            layer_kv += (korv[:, :, : prev_input_len + tgt.shape[-2], :],)
        new_kv += (layer_kv,)

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]

    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]

    draft_tokens, retrieve_indices,tree_mask,tree_position_ids = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head,logits_processor=logits_processor)


    new_token += accept_length + 1

    return input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, None, token, new_kv


def eagenerate(
            model,
            input_ids,
            temperature=0.0,
            top_p=0.0,
            top_k=0.0,
            max_new_tokens=512,
            max_length=2048,
            log=False,
            is_llama3=False,

    ):
        if is_llama3:
            stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        max_length=max_length-model.ea_layer.total_tokens-10

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature, top_p=top_p, top_k=top_k)
        else:
            logits_processor = None
        #assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place

        padding=(torch.zeros(1,1,dtype=torch.long)-1).to(input_ids.device)
        input_ids = input_ids.clone()
        model.ea_layer.reset_kv()

        input_len = input_ids.shape[1]
        # with Timer("initialize_tree"):
        draft_tokens, retrieve_indices,tree_mask,tree_position_ids, logits, hidden_state, sample_token, past_key_values = initialize_tree(
            input_ids, model, logits_processor
        )

        new_token = 0

        for idx in range(max_length):
            # with Timer("all"):
            draft_tokens=draft_tokens.to(input_ids.device)
            #with Timer("tree_decoding"):
            logits, hidden_state_new, past_key_values = tree_decoding(
                        model,
                        draft_tokens,
                        past_key_values,
                        tree_position_ids,
                        input_ids,
                        retrieve_indices,
                        tree_mask=tree_mask,
                    )

            draft_tokens=torch.cat((draft_tokens,padding),dim=1)
            candidates=draft_tokens[0,retrieve_indices]
            # with Timer("evaluate_posterior"):
            best_candidate, accept_length, sample_p = evaluate_posterior(
                logits, candidates, logits_processor
            )

            # print("new_token: ", (accept_length+1).item())
            # with Timer("update_inference_inputs"):
            input_ids, draft_tokens, retrieve_indices,tree_mask,tree_position_ids, new_token, hidden_state, sample_token, past_key_values = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values,
                # current_length_data,
                model,
                hidden_state_new,
                sample_p
            )

            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break

            if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > max_length:
                break
        if not log:
            return input_ids
        else:
            return input_ids, new_token, idx


def run_eval(
        base_model_path,
        ea_model_path,
        model_id,
        question_file,
        question_begin,
        question_end,
        answer_file,
        max_new_token,
        num_choices,
        temperature,
        enable_ipex_llm,
        args,
):
    questions = load_questions(question_file, question_begin, question_end)
    shuffled_ids = [q["question_id"] for q in questions]

    get_answers_func = get_model_answers

    chunk_size = len(questions)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                base_model_path,
                ea_model_path,
                model_id,
                questions[i: i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                temperature,
                enable_ipex_llm,
                args
            )
        )


@torch.inference_mode()
def get_model_answers(
        base_model_path,
        ea_model_path,
        model_id,
        questions,
        answer_file,
        max_new_token,
        num_choices,
        temperature,
        enable_ipex_llm,
        args
):

    model = EaModel.from_pretrained(
            base_model_path=base_model_path,
            ea_model_path=ea_model_path,
            torch_dtype=torch.float16,
            # torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            total_token=args.total_token,
            depth=args.depth,
            top_k=args.top_k,

        )
    if enable_ipex_llm:
        # Use optimized int4 base model to replace base model in EaModel
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path, load_in_4bit=True, use_cache=True,
                                                          torch_dtype=torch.float16)
        model.base_model = base_model
        # Also optimize draft model in EaModel
        model.ea_layer = optimize_model(model.ea_layer)
    model = model.to("xpu")
    model.ea_layer.tree_mask_init = model.ea_layer.tree_mask_init.to("xpu")
    model.ea_layer.position_ids = model.ea_layer.position_ids.to("xpu")
    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()

    print('Check model training state:', model.training)


    question = questions[0]

    # warmup
    for _ in range(3):
        torch.manual_seed(0)

        conv = get_conversation_template("llama-2-chat")
        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        conv.system_message = sys_p
        turns = []
        idxs = []
        new_tokens = []
        wall_time = []
        fail_count = 0
        for j in range(len(question["turns"])):
            qs = question["turns"][j]
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt() + " "
            inputs = tokenizer([prompt], return_tensors="pt").to("xpu")
            input_ids = inputs.input_ids

            torch.xpu.synchronize()
            start_time = time.time()

            output_ids, new_token, idx = eagenerate(
                model,
                torch.as_tensor(input_ids),
                temperature=temperature,
                max_new_tokens=max_new_token,
                log=True
            )
            torch.xpu.synchronize()
            total_time = time.time() - start_time
            output_ids = output_ids[0][len(input_ids[0]):]
            # be consistent with the template's stop_token_ids
            if conv.stop_token_ids:
                stop_token_ids_index = [
                    i
                    for i, id in enumerate(output_ids)
                    if id in conv.stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )
            conv.stop_str = "</s>"
            if conv.stop_str and output.find(conv.stop_str) > 0:
                output = output[: output.find(conv.stop_str)]
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            if conv.name == "xgen" and output.startswith("Assistant:"):
                output = output.replace("Assistant:", "", 1).strip()
            turns.append(output)
            idxs.append(int(idx))
            new_tokens.append(int(new_token))
            wall_time.append(total_time)
            conv.messages[-1][-1] = output

    for question in tqdm(questions):

        choices = []
        for i in range(num_choices):
            torch.manual_seed(i)
            conv = get_conversation_template("llama-2-chat")
            sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
            conv.system_message = sys_p
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            for j in range(len(question["turns"])):
                qs = question["turns"][j]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt() + " "
                inputs = tokenizer([prompt], return_tensors="pt").to("xpu")
                input_ids = inputs.input_ids

                try:
                    torch.xpu.synchronize()
                    start_time = time.time()
                    output_ids, new_token, idx = eagenerate(
                        model,
                        torch.as_tensor(input_ids),
                        temperature=temperature,
                        max_new_tokens=max_new_token,
                        log=True,
                    )
                    torch.xpu.synchronize()
                    total_time = time.time() - start_time
                    output_ids = output_ids[0][len(input_ids[0]):]

                    if conv.stop_token_ids:
                        stop_token_ids_index = [
                            i
                            for i, id in enumerate(output_ids)
                            if id in conv.stop_token_ids
                        ]
                        if len(stop_token_ids_index) > 0:
                            output_ids = output_ids[: stop_token_ids_index[0]]

                    output = tokenizer.decode(
                        output_ids,
                        spaces_between_special_tokens=False,
                    )
                    if conv.stop_str and output.find(conv.stop_str) > 0:
                        output = output[: output.find(conv.stop_str)]
                    for special_token in tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output = output.replace(special_tok, "")
                        else:
                            output = output.replace(special_token, "")
                    output = output.strip()

                    if conv.name == "xgen" and output.startswith("Assistant:"):
                        output = output.replace("Assistant:", "", 1).strip()
                except RuntimeError as e:
                    print("ERROR question ID: ", question["question_id"])
                    output = "ERROR"

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                conv.messages[-1][-1] = output
            choices.append({"index": i, "turns": turns, "idxs": idxs, "new_tokens": new_tokens, "wall_time": wall_time})

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ea-model-path",
        type=str,
        default="/home/lyh/weights/hf/eagle/llama2chat/7B/",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--base-model-path", type=str, default="/home/lyh/weights/hf/llama2chat/7B/",
                        help="1")
    parser.add_argument("--model-id", type=str, default="ess-llama-2-chat-7b-fp32")
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=8,
        help="The total number of nodes in the draft tree",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )

    parser.add_argument(
        "--enable-ipex-llm",
        action='store_true',
        help="Enable ipex-llm optimization"
    )
    args = parser.parse_args()

    args.model_id = args.model_id + "-temperature-" + str(args.temperature)
    # args.tree_choices = eval(args.tree_choices)

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        args.base_model_path,
        args.ea_model_path,
        args.model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.temperature,
        args.enable_ipex_llm,
        args
    )

    reorg_answer_file(answer_file)

