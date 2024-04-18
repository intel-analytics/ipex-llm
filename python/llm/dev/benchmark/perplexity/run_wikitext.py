import argparse
import torch
from datasets import load_dataset
from tqdm import tqdm


def parse_kwargs(kwstr):
    kvpair = [item.split('=') for item in kwstr.split(',') if item != ""]
    return {k:v for k, v in kvpair}


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str)
parser.add_argument("--dataset", type=str, default='path=wikitext,name=wikitext-2-raw-v1')
parser.add_argument("--device", type=str, default="xpu")
parser.add_argument("--precision", type=str, default="sym_int4")
parser.add_argument("--use-cache", action="store_true")
parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples per task. For debug only")
args = parser.parse_args()

if args.precision == "fp16":  # ipex fp16
    from transformers import AutoModelForCausalLM
    if "xpu" in args.device:
        import intel_extension_for_pytorch as ipex
    model = AutoModelForCausalLM.from_pretrained(args.model_path, use_cache=args.use_cache, trust_remote_code=True)
    model = model.half()
else:  # ipex-llm
    from ipex_llm.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_low_bit=args.precision,
                                                 use_cache=args.use_cache, trust_remote_code=True)
    model = model.half()
model = model.to(args.device)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

test = load_dataset(**parse_kwargs(args.dataset), split="test")["text"]
if args.limit:
    test = test[:args.limit]
encodings = tokenizer("\n\n".join(test), return_tensors="pt")

max_length = model.config.max_position_embeddings
stride = 512
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)
    if "xpu" in args.device:
        torch.xpu.empty_cache()

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print("ppl result: {}".format(ppl.item()))
