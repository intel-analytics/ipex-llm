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


# this code is copied from llama2 example test, and added performance test
import argparse
import os
import subprocess

task_cmd = "--num_fewshot {} --tasks {}"

task_map = {
    "hellaswag": task_cmd.format(10, "hellaswag"),
    "arc": task_cmd.format(25, "arc_challenge"),
    "truthfulqa": task_cmd.format(0, "truthfulqa_mc"),
    "mmlu": task_cmd.format(5, "hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions")
}

prec_to_arg = {
    "bigdl-llm": {
        "int4": "load_in_low_bit=sym_int4",
        "nf4": "load_in_low_bit=nf4",
        "nf3": "load_in_low_bit=nf3",
        "fp8": "load_in_low_bit=fp8",
        "fp4": "load_in_low_bit=fp4",
        "bf16": "dtype=bfloat16",
        "fp16": "dtype=float16",
    },
    "hf-causal": {
        "nf4": "bnb_type=nf4",
        "bf16": "dtype=bfloat16",
        "fp16": "dtype=float16",
    }
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--pretrained", required=True, type=str)
    parser.add_argument("--precision", required=True, nargs='+', type=str)
    parser.add_argument("--device", required=True, type=str)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--tasks", required=True, nargs='+', type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    print(args.model)
    print(args.tasks)
    basic_cmd = "python lm-evaluation-harness/main.py --model {} --model_args pretrained={},{} --no_cache --device {} --batch_size {} {} --output_path {} "
    os.makedirs(args.output_dir, exist_ok=True)
    index = 1
    total = len(args.precision) * len(args.tasks)
    for prec in args.precision:
        prec_arg = prec_to_arg[args.model][prec]
        for task in args.tasks:
            output_path = f"{args.model}_{prec}_{args.device}_{task}"
            task_arg = task_map[task]
            cmd_exec = basic_cmd.format(args.model, args.pretrained, prec_arg, args.device, args.batch,
             task_arg, f"{args.output_dir}/{output_path}")
            print(f"Running job {index}/{total}:\n{cmd_exec}")
            index += 1
            with open(f"{args.output_dir}/log_{output_path}.txt", "w") as f:
                return_code = subprocess.call(cmd_exec, shell=True, stderr=f, stdout=f)
            if return_code == 0:
                print("Successful")
            else:
                print("Failed")

main()
