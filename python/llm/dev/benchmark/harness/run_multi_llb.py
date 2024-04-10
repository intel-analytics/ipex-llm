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
import argparse
import json
import logging
import os
from harness_to_leaderboard import *
from lm_eval import tasks, evaluator, utils, models
from multiprocessing import Queue, Process
import multiprocessing as mp
from contextlib import redirect_stdout, redirect_stderr

from ipexllm import IPEXLLM
models.MODEL_REGISTRY['ipex-llm'] = IPEXLLM    # patch ipex-llm to harness

logging.getLogger("openai").setLevel(logging.WARNING)


def parse_device(device):
    device = device.split(':')
    if len(device) == 0:
        return device
    device_indices = device[1].split(',')
    return list(map(lambda i: f"{device[0]}:{i}", device_indices))

def run_job(device, prec, task, args, device_pool, result_pool):
    print(f"Current Job: device={device}, precision={prec}, task={task}")
    device_type = device.split(':')[0]
    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    model_name = os.path.basename(os.path.realpath(args.pretrained))
    output_path = args.output_path if args.output_path else "results"
    
    prec_arg = parse_precision(prec, args.model)
    model_args = f"pretrained={args.pretrained},{prec_arg}"
    if len(args.model_args) > 0:
        model_args = f"{model_args},{args.model_args}"
    task_names=task_map.get(task, task).split(',')
    num_fewshot = task_to_n_few_shots.get(task, args.num_fewshot)
    log_dir = f"{output_path}/{model_name}/{device_type}/{prec}/{task}"
    os.makedirs(log_dir, exist_ok=True)

    with open(f"{log_dir}/log.txt", 'w') as f, redirect_stderr(f), redirect_stdout(f):
        results = evaluator.simple_evaluate(
            model=args.model,
            model_args=model_args,
            tasks=task_names,
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            device=device,
            no_cache=args.no_cache,
            limit=args.limit,
            description_dict=description_dict,
            decontamination_ngrams_path=args.decontamination_ngrams_path,
            check_integrity=args.check_integrity,
            write_out=args.write_out,
            output_base_path=log_dir
        )
        if len(results['results']) > 1:
            average = {}
            for _, subtask in results['results'].items():
                for metric, value in subtask.items():
                    average[metric] = average.get(metric, []) + [value]
            for k, v in average.items():
                average[k] = sum(v) / len(v) if not k.endswith("_stderr") else 0
            results['results'][task] = average
            results['versions'][task] = 1

        dumped = json.dumps(results, indent=2)
        print(dumped)

        if args.output_path:
            with open(f"{log_dir}/result.json", "w") as f:
                f.write(dumped)
        result_pool.put(results)
        device_pool.put(device)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--pretrained", required=True, type=str)
    parser.add_argument("--tasks", required=True, nargs='+', type=str)
    parser.add_argument("--precision", required=True, nargs='+', type=str)
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default=None)
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        help="Maximal batch size to try with --batch_size auto",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=False)
    parser.add_argument("--output_base_path", type=str, default=None)

    return parser.parse_args()


def main():
    mp.set_start_method('spawn')
    args = parse_args()
    
    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    print(f"Selected Tasks: {args.tasks}")

    device_pool = Queue()
    result_pool = Queue()
    for device in parse_device(args.device):
        device_pool.put(device)

    jobs = []
    for prec in args.precision:
        for task in args.tasks:
            device = device_pool.get()
            p = Process(target=run_job, args=(device, prec, task, args, device_pool, result_pool))
            p.start()
            jobs.append(p)

    for j in jobs:
        j.join()
    
    while not result_pool.empty():
        result = result_pool.get()
        print(result if isinstance(result, str) else evaluator.make_table(result))

if __name__ == "__main__":
    main()
