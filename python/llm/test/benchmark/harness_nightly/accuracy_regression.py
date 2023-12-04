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
import json
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(res_path, golden_path):
    print(res_path, golden_path)
    with open(res_path, "r") as f:
        results = json.load(f)['results']
        print(results)
    
    model_name, device, precision, task = res_path.split('/')[-5:-1]

    with open(golden_path, "r") as f:
        golden_results = json.load(f)[model_name][device][precision]
        print(golden_results)

    identical = True
    for task in results.keys():

        if task not in golden_results:
            identical = False
            logger.error(f"Task {task} should be updated to golden results.")
            continue
        task_results = results[task]
        task_golden = golden_results[task]
        for m in task_results.keys():
            if m in task_golden and abs(task_results[m] - task_golden[m]) < 0.001:
                if not m.endswith("_stderr"):
                    identical = False
                    logger.error(f"Different on metric '{m}' [golden acc/ current acc]: [{task_golden[m]}/{task_results[m]}]")
                else:
                    logger.warning(f"Diff on {m} [golden acc/ current acc]: [{task_golden[m]}/{task_results[m]}]")
    if identical:
        logger.info("Accuracy values are identical to golden results.")
    else:
        raise RuntimeError("Accuracy has changed, please check if any accuracy issue or update golden accuracy value.")

main(*sys.argv[1:3])