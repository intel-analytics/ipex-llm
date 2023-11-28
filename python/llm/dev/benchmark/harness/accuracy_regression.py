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
            if m in task_golden and task_results[m] != task_golden[m]:
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