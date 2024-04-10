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
from regex import match


task_map = dict(
    hellaswag="hellaswag",
    arc="arc_challenge",
    truthfulqa="truthfulqa_mc",
    mmlu="hendrycksTest-abstract_algebra,hendrycksTest-anatomy,hendrycksTest-astronomy,hendrycksTest-business_ethics,hendrycksTest-clinical_knowledge,hendrycksTest-college_biology,hendrycksTest-college_chemistry,hendrycksTest-college_computer_science,hendrycksTest-college_mathematics,hendrycksTest-college_medicine,hendrycksTest-college_physics,hendrycksTest-computer_security,hendrycksTest-conceptual_physics,hendrycksTest-econometrics,hendrycksTest-electrical_engineering,hendrycksTest-elementary_mathematics,hendrycksTest-formal_logic,hendrycksTest-global_facts,hendrycksTest-high_school_biology,hendrycksTest-high_school_chemistry,hendrycksTest-high_school_computer_science,hendrycksTest-high_school_european_history,hendrycksTest-high_school_geography,hendrycksTest-high_school_government_and_politics,hendrycksTest-high_school_macroeconomics,hendrycksTest-high_school_mathematics,hendrycksTest-high_school_microeconomics,hendrycksTest-high_school_physics,hendrycksTest-high_school_psychology,hendrycksTest-high_school_statistics,hendrycksTest-high_school_us_history,hendrycksTest-high_school_world_history,hendrycksTest-human_aging,hendrycksTest-human_sexuality,hendrycksTest-international_law,hendrycksTest-jurisprudence,hendrycksTest-logical_fallacies,hendrycksTest-machine_learning,hendrycksTest-management,hendrycksTest-marketing,hendrycksTest-medical_genetics,hendrycksTest-miscellaneous,hendrycksTest-moral_disputes,hendrycksTest-moral_scenarios,hendrycksTest-nutrition,hendrycksTest-philosophy,hendrycksTest-prehistory,hendrycksTest-professional_accounting,hendrycksTest-professional_law,hendrycksTest-professional_medicine,hendrycksTest-professional_psychology,hendrycksTest-public_relations,hendrycksTest-security_studies,hendrycksTest-sociology,hendrycksTest-us_foreign_policy,hendrycksTest-virology,hendrycksTest-world_religions",
    winogrande='winogrande',
    gsm8k='gsm8k',
    drop='drop'
)


task_to_n_few_shots = dict(
    hellaswag=10,
    arc=25,
    truthfulqa=0,
    mmlu=5,
    winogrande=5,
    gsm8k=5,
    drop=3
)


task_to_metric = dict(
    hellaswag='acc_norm',
    arc='acc_norm',
    truthfulqa='mc2',
    mmlu='acc',
    winogrande='acc',
    gsm8k='acc',
    drop='f1'
)

def parse_precision(precision, model="ipex-llm"):
    result = match(r"([a-zA-Z_]+)(\d+)([a-zA-Z_\d]*)", precision)
    datatype = result.group(1)
    bit = int(result.group(2))
    if bit >= 16:
        float_map = dict(
            bf16="bfloat16",
            fp16="float16",
            fp32="float32"
        )
        return f"dtype={float_map[precision]}"
    else:
        if model == "hf-causal":
            return f"bnb_type={precision}"
        if model == "ipex-llm":
            return f"load_in_low_bit={precision}"
    raise RuntimeError(f"invald precision {precision}")    
