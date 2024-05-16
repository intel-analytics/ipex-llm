import argparse
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", required=True, type=str)
parser.add_argument("--data_path", type=str, default='wikitext-2-raw-v1/wikitext-2-raw/wiki.test.raw')
parser.add_argument("--chunk_size", type=int, default=512)
parser.add_argument("--stride", type=int, default=0)
parser.add_argument("--device", type=str, default="xpu")
parser.add_argument("--precision", type=str, default="sym_int4")
parser.add_argument("--use-cache", action="store_true")
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

with open(args.data_path, "rb") as f:
    data = f.read()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
encodings = tokenizer(data.decode("utf-8").strip("\n"), return_tensors="pt")

max_length = model.config.max_position_embeddings
stride = args.chunk_size if args.stride <= 0 else args.stride
seq_len = encodings.input_ids.size(1)
num_chunks = seq_len // stride

nlls = []
prev_end_loc = 0
for i in tqdm(range(num_chunks)):
    begin_loc = i * stride
    if args.stride > 0:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    else:
        end_loc = begin_loc + stride
        trg_len = -stride//2
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(args.device)
    if args.stride == 0: input_ids[:, 0] = tokenizer.bos_token_id
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
print("Final ppl estimate: {}".format(ppl.item()))
