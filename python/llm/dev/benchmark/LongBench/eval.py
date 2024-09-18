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
# https://github.com/THUDM/LongBench/blob/main/eval.py
# and
# https://github.com/FasterDecoding/SnapKV/blob/main/experiments/LongBench/eval.py

import os
import json
import argparse
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)


def result_path_range(full_kv: bool, configs: list[str], model_name: str, fa_name: str):
    if full_kv:
        yield f"{fa_name}/{model_name}"

    for config in configs:
        yield f"{fa_name}/{model_name}_{config}"


if __name__ == '__main__':
    from omegaconf import OmegaConf
    conf = OmegaConf.load(f'{current_dir}/config.yaml')

    model_names = conf['model_name'] if OmegaConf.is_list(conf['model_name']) else [conf['model_name']]
    full_kv = conf['full_kv']
    ees = conf['e'] if OmegaConf.is_list(conf['e']) else [conf['e']]
    compresskv_configs = conf['compress_kv'] if OmegaConf.is_list(conf['compress_kv']) else [conf['compress_kv']]

    model2maxlen = json.load(open(f"{current_dir}/config/model2maxlen.json", "r"))

    for model_name in model_names:
        max_length = model2maxlen[model_name]
        for e in ees:
            fa_dir_name = f"pred_{'e_' if e else ''}{max_length}"
            for path in result_path_range(full_kv, compresskv_configs, model_name, fa_dir_name):
                scores = dict()
                all_files = os.listdir(path)
                print("Evaluating on:", all_files)
                for filename in all_files:
                    if not filename.endswith("jsonl"):
                        continue
                    predictions, answers, lengths = [], [], []
                    dataset = filename.split('.')[0]
                    with open(f"{path}/{filename}", "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            predictions.append(data["pred"])
                            answers.append(data["answers"])
                            all_classes = data["all_classes"]
                            if "length" in data:
                                lengths.append(data["length"])
                    if e:
                        score = scorer_e(dataset, predictions, answers, lengths, all_classes)
                    else:
                        score = scorer(dataset, predictions, answers, all_classes)
                        if dataset == 'qasper':
                            score_e = scorer_e(dataset, predictions, answers, lengths, all_classes)
                    scores[dataset] = score

                out_path = f"{path}/result.json"
                with open(out_path, "w") as f:
                    json.dump(scores, f, ensure_ascii=False, indent=4)
