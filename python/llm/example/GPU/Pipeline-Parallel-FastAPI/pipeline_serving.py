from llama_models import ModelRunner
import torch.nn.parallel
import torch.distributed as dist
import os
import intel_extension_for_pytorch as ipex

import oneccl_bindings_for_pytorch

from transformers.utils import logging
logger = logging.get_logger(__name__)

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29501'

backend = 'ccl'
dist.init_process_group(backend)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
device = f"xpu:{my_rank}"
logger.info(f"rank: {my_rank}, size: {my_size}")

import time
from transformers import AutoTokenizer, AutoConfig


# checkpoint = "EleutherAI/gpt-j-6B"
# weights_location = snapshot_download(repo_id=checkpoint, allow_patterns="pytorch*.*")
checkpoint = "/mnt/disk1/models/Llama-2-13b-chat-hf/"

# serialize model initialization so that we do not run out of CPU memory
if my_rank == 0:
    logger.info("start model initialization")
    local_model = ModelRunner(checkpoint, my_rank, my_size)
    logger.info("model initialized")
dist.barrier()
if my_rank == 1:
    logger.info("start model initialization")
    local_model = ModelRunner(checkpoint, my_rank, my_size)
    logger.info("model initialized")
dist.barrier()

# torch ccl requires a collective operation first before send and recive
x = torch.ones([1, 10], device=device, dtype=torch.float16) * my_rank
logger.info("before first allreduce")
dist.all_reduce(x)

logger.info("successful run fist all reduce")

tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
input_str_32 = "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun"
input_str_1024 = "It is done, and submitted. You can play 'Survival of the Tastiest' on Android, and on the web. Playing on the web works, but you have to simulate multiple touch for table moving and that can be a bit confusing. There is a lot I'd like to talk about. I will go through every topic, insted of making the typical what went right/wrong list. Concept Working over the theme was probably one of the hardest tasks which I had to face. Originally, I had an idea of what kind of game I wanted to develop, gameplay wise - something with a lot of enemies/actors, simple graphics, maybe set in space, controlled from a top-down view. I was confident that I could fit any theme around it. In the end, the problem with a theme like 'Evolution' in a game is that evolution is unassisted. It happens through several seemingly random mutations over time, with the most apt permutation surviving. This genetic car simulator is, in my opinion, a great example of actual evolution of a species facing a challenge. But is it a game? In a game, you need to control something to reach an objective. That control goes against what evolution is supposed to be like. If you allow the user to pick how to evolve something, it's not evolution anymore - it's the equivalent of intelligent design, the fable invented by creationists to combat the idea of evolution. Being agnostic and a Pastafarian, that's not something that rubbed me the right way. Hence, my biggest dillema when deciding what to create was not with what I wanted to create, but with what I did not. I didn't want to create an 'intelligent design' simulator and wrongly call it evolution. This is a problem, of course, every other contestant also had to face. And judging by the entries submitted, not many managed to work around it. I'd say the only real solution was through the use of artificial selection, somehow. So far, I haven't seen any entry using this at its core gameplay. Alas, this is just a fun competition and after a while I decided not to be as strict with the game idea, and allowed myself to pick whatever I thought would work out. My initial idea was to create something where humanity tried to evolve to a next level, but had some kind of foe trying to stop them from doing so. I kind of had this image of human souls flying in space towards a monolith or a space baby (all based in 2001: A Space Odyssey of course) but I couldn't think of compelling (read: serious) mechanics for that. Borgs were my next inspiration, as their whole hypothesis fit pretty well into the evolution theme. But how to make it work? Are you the borg, or fighting the Borg? The third and final idea came to me through my girlfriend, who somehow gave me the idea of making something about the evolution of Pasta. The more I thought about it the more it sounded like it would work, so I decided to go with it. Conversations with my inspiring co-worker Roushey (who also created the 'Mechanical Underdogs' signature logo for my intros) further matured the concept, as it involved into the idea of having individual pieces of pasta flying around and trying to evolve until they became all-powerful. A secondary idea here was that the game would work to explain how the Flying Spaghetti Monster came to exist - by evolving from a normal dinner table. So the idea evolved more or less into this: you are sitting a table. You have your own plate, with is your 'base'. There are 5 other guests at the table, each with their own plate. Your plate can spawn little pieces of pasta. You do so by 'ordering' them through a menu. Some pastas are better than others; some are faster, some are stronger. They have varying 'costs', which are debited from your credits (you start with a number of credits). Once spawned, your pastas start flying around. Their instinct is to fly to other plates, in order to conquer them (the objective of the game is having your pasta conquer all the plates on the table). But they are really autonomous, so after being spawned, you have no control over your pasta (think DotA or LoL creeps). Your pasta doesn't like other people's pasta, so if they meet, they shoot sauce at each other until one dies. You get credits for other pastas your own pasta kill. Once a pasta is in vicinity of a plate"

input_ids = tokenizer.encode(input_str_32, return_tensors="pt").to(device)

for i in range(3):
    output_ids = local_model.generate(input_ids, max_tokens=32)

logger.info(tokenizer.decode(output_ids[0]))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio, uuid
from typing import Dict, List, Optional
import argparse

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return int(default)


class PromptRequest(BaseModel):
    prompt: str
    n_predict: int = 32

empty_req = PromptRequest(prompt="", n_predict=0)

app = FastAPI()

request_queue: asyncio.Queue = asyncio.Queue()
result_dict: Dict[str, str] = {}
local_rank = my_rank
max_num_seqs = get_int_from_env(["MAX_NUM_SEQS"], "16")


@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    request_id = str(uuid.uuid4())
    await request_queue.put((request_id, prompt_request))
    while True:
        await asyncio.sleep(0.1)
        if request_id in result_dict:
            output_str = result_dict.pop(request_id)
            return {"generated_text": output_str}


def generate_text(prompt: List[str], n_predict = 32):
    while prompt[-1] == "":
        prompt = prompt[:-1]
    if isinstance(n_predict, list):
        n_predict = max(n_predict)
        
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids.to(f'xpu:{local_rank}')
    print(inputs)
    attention_mask = inputs.attention_mask.to(f'xpu:{local_rank}')
    output = local_model.generate(input_ids,
                                  max_tokens=n_predict,
                            # attention_mask=attention_mask,
                            # max_new_tokens=n_predict,
                            # min_new_tokens=n_predict,
                            # do_sample=False,
                            # use_cache=True
                            )
    torch.xpu.synchronize()

    return output


async def process_requests():
    while True:
        request_ids, prompt_requests = [], []
        for _ in range(max_num_seqs):
            if request_queue.empty():
                break
            request_id, prompt_request = await request_queue.get()
            request_ids.append(request_id)
            prompt_requests.append(prompt_request)

        if local_rank == 0 and prompt_requests:
            # import pdb
            # pdb.set_trace()
            object_list = prompt_requests
            if len(object_list) < max_num_seqs:
                object_list = object_list + [empty_req] * (max_num_seqs - len(object_list))
            logger.info(f"Running: {len(prompt_requests)}, Pending: {request_queue.qsize()}")
            dist.broadcast_object_list(object_list, src=0)
            start_time = time.time()
            outputs = generate_text([req.prompt for req in object_list], [req.n_predict for req in object_list])
            # print(outputs)
            generate_time = time.time() - start_time
            outputs = outputs.cpu()
            output_strs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            output_strs = output_strs[:len(prompt_requests)]

            for request_id, output_str in zip(request_ids, output_strs):
                result_dict[request_id] = output_str
            # print(result_dict)
            # logger.info(f"Token latency: {result[-1]}, generate time: {generate_time}")

        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    if local_rank == 0:
        asyncio.create_task(process_requests())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Tokens using fastapi by leveraging DeepSpeed-AutoTP')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Llama-2-13b-chat-hf` and `meta-llama/Llama-2-70b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--low-bit', type=str, default='sym_int4',
                    help='The quantization type the model will convert to.')
    parser.add_argument('--port', type=int, default=8000,
                    help='The port number on which the server will run.')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    low_bit = args.low_bit
    # load_model(model_path, low_bit)
    if local_rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        while True:
            object_list = [None] * max_num_seqs
            dist.broadcast_object_list(object_list, src=0)
            output = generate_text([req.prompt for req in object_list], [req.n_predict for req in object_list])
