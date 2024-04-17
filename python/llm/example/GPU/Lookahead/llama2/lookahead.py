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

import torch
from ipex_llm.transformers import AutoModelForCausalLM
from transformers import LlamaTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
LLAMA2_PROMPT_FORMAT = """[INST] {prompt} [/INST]"""

daily_text = """(CNN) -- A former government contract employee was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk, the Department of Justice announced Thursday. Sources say the classified materials were taken from the East Tennessee Technology Park. Roy Lynn Oakley, 67, of Roane County, Tennessee, appeared in federal court in Knoxville on Thursday. Oakley was briefly detained for questioning in the case in January, when authorities first learned of the alleged plot to divulge the materials, government sources told CNN. He voluntarily surrendered Thursday at an FBI field office in Knoxville, the sources said. Oakley is a former employee of Bechtel Jacobs, the Department of Energy's prime environmental management contractor at the East Tennessee Technology Park, prosecutors said. The indictment states that Oakley, "having possession of, access to and having been entrusted with sections of 'barriers' and associated hardware used for uranium enrichment through the process of gaseous diffusion ... having reason to believe that such data would be utilized to injure the United States and secure an advantage to a foreign nation, did communicate, transmit and disclose such data to another person." The transfer took place January 26, the indictment alleges. Oakley is also charged with converting the material and "restricted data" to his own use. He began doing so on about October 17, 2006, and continued through January, prosecutors said. Prosecutors said the materials involved have been examined by scientists and posed no threat to people who may have come into contact with them. Oakley's attorney, Herb Moncier, said outside court Thursday that Oakley's job was to break rods "into little pieces" and throw them away. Moncier said Oakley had a security clearance, but Moncier did not believe it was a high-level clearance. The government alleges that in January, Oakley attempted to sell the "pieces of scrap" to someone he thought was a French agent -- but in reality was an undercover FBI agent, Moncier said. He said he questions whether those broken pieces would be considered an "appliance" under the law. "Mr. Oakley has cooperated fully for the last six months," said Moncier, who added that he had traveled to Washington for work on the case. Each count carries a possible sentence upon conviction of up to 10 years in prison and a $250,000 fine. "While none of the stolen equipment was ever transmitted to a foreign government or terrorist organization, the facts of this case demonstrate the importance of safeguarding our nuclear technology and pursuing aggressive prosecution against those who attempt to breach the safeguards and put that technology in the wrong hands," Kenneth Wainstein, assistant attorney general for national security, said in the Justice Department statement. One government source said the materials involved are not the "crown jewels," but they should not have been taken from the facility. A "barrier" is used to filter uranium during the enrichment process, according to nuclear energy officials, but a significant number of barriers are needed to do that job. Sources told CNN that federal authorities have been following Oakley and investigating the case for at least six months, after he allegedly tried to sell the classified material. Oakley, described as a low-level employee, apparently did not make contact with any foreign government and is not a foreign agent of any kind, an official familiar with the case said. A government official with with knowledge of the case said that when authorities learned of Oakley's alleged intentions six months ago, the FBI and Department of Energy launched a joint investigation. The FBI then developed a sting operation, government officials familiar with the case said, and authorities intervened before there could be any involvement of a foreign country. East Tennessee Technology Park is an area of the DOE's Oak Ridge reservation "where we are currently decontaminating and decommissioning buildings that were last used in 1985," Gerald Boyd, manager of the DOE's Oak Ridge site office, said Thursday. "When they were in use, now over 20 years ago, some of the buildings at ETTP housed facilities used for the enrichment of uranium." Boyd said the technology park and the reservation "are protected by multiple layers of security systems and detection programs, both visible and unseen, meant to identify rogue employees attempting to abuse their access and position." In this case, a review of security procedures showed that the system worked and "successfully identified the individual in question," he said. E-mail to a friend . CNN's Terry Frieden and Kelli Arena contributed to this report."""
question = "Can you please summarize this article?"
long_input = "Article:```{daily_text}``` \n\n Question: {question} \n\n".format(daily_text=daily_text, question=question)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--prompt', type=str, default=long_input,
                        help='Prompt to infer')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--n-lookahead-tokens', type=int, default=3,
                        help='Number of tokens to lookahead')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    # Load model in optimized sym_int4 here.
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 load_in_4bit=True,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    model = model.to('xpu')

    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    with torch.inference_mode():
        prompt = LLAMA2_PROMPT_FORMAT.format(prompt=args.prompt)
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(model.device)

        # warmup
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                lookahead=args.n_lookahead_tokens,
                                do_sample=False)
        output_str = tokenizer.decode(output[0])

        # lookahead generate
        st = time.perf_counter()
        output = model.generate(input_ids,
                                max_new_tokens=args.n_predict,
                                lookahead=args.n_lookahead_tokens,
                                do_sample=False)
        output_str = tokenizer.decode(output[0], skip_special_tokens=True)
        torch.xpu.synchronize()
        end = time.perf_counter()

        print(output_str)
        print(f"Tokens generated {model.n_token_generated}")
        print(f"E2E Generation time {(end - st):.4f}s")
        print(f"First token latency {model.first_token_time:.4f}s")
