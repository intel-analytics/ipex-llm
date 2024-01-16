from ppl import BigDLPPL
from datasets import load_dataset
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    parser.add_argument("--low_bit", required=False, type=str, default=None, nargs='+')
    parser.add_argument("--model_kwargs", required=False, type=str, default={}, nargs='+')
    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    text = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    for low_bit in args.low_bit:
        ppl_evaluator = BigDLPPL(model_path=args.model_path, device=args.device, **args.model_kwargs)
        ppl = ppl_evaluator.perplexity_hf(text)
        print(f'{low_bit}:{ppl}')


main()