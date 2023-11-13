import argparse
import json
import logging
import os
from harness_to_leaderboard import *
from lm_eval import tasks, evaluator, utils, models

from bigdl_llm import BigDLLM
models.MODEL_REGISTRY['bigdl-llm'] = BigDLLM    # patch bigdl-llm to harness

logging.getLogger("openai").setLevel(logging.WARNING)


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
    args = parse_args()

    assert not args.provide_description  # not implemented

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    
    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = utils.pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {args.tasks}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    summary = []
    for prec in args.precision:
        prec_arg = parse_precision(prec, args.model)
        model_args = f"pretrained={args.pretrained},{prec_arg}"
        if len(args.model_args) > 0:
            model_args += args.model_args
        for task in args.tasks:
            task_names=task_map.get(task, task).split(',')
            num_fewshot = task_to_n_few_shots.get(task, args.num_fewshot)
            try:
                results = evaluator.simple_evaluate(
                    model=args.model,
                    model_args=model_args,
                    tasks=task_names,
                    num_fewshot=num_fewshot,
                    batch_size=args.batch_size,
                    max_batch_size=args.max_batch_size,
                    device=args.device,
                    no_cache=args.no_cache,
                    limit=args.limit,
                    description_dict=description_dict,
                    decontamination_ngrams_path=args.decontamination_ngrams_path,
                    check_integrity=args.check_integrity,
                    write_out=args.write_out,
                    output_base_path=args.output_base_path,
                )
                if len(results['results']) > 1:
                    average = {}
                    for _, subtask in results['results'].items():
                        for metric, value in subtask.items():
                            average[metric] = average.get(metric, []) + [value]
                    for k, v in average.items():
                        average[k] = sum(average[k]) / len(average[k]) if not k.endswith("_stderr") else 0
                    results['results'][f"avg_{task}"] = average

                dumped = json.dumps(results, indent=2)
                print(dumped)

                if args.output_path:
                    dirname = os.path.dirname(args.output_path)
                    if dirname:
                        os.makedirs(dirname, exist_ok=True)
                    with open(args.output_path, "w") as f:
                        f.write(dumped)

                batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
                print(
                    f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
                    f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
                )
                print(evaluator.make_table(results))
                summary.append(results)
            except Exception as e:
                print(f"Job config of task={task}, precision={prec} failed. Error Message: {str(e)}")
    
    ## print all task summary
    for results in summary:
        print(evaluator.make_table(results))

if __name__ == "__main__":
    main()
