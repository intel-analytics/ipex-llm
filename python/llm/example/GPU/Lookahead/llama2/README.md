# LLaMA2
In this directory, you will find examples on how you could run LLaMA2 inference with lookahead optimization using IPEX-LLM on [Intel GPUs](../README.md). For illustration purposes, we utilize the [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf) as reference Llama2 models.

## 0. Requirements
To run these examples with IPEX-LLM on Intel GPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API
In the example [lookahead.py](./lookahead.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM lookahead optimizations on Intel GPUs.
### 1. Install
We suggest using conda to manage environment:
```bash
conda create -n llm python=3.11
conda activate llm
# below command will install intel_extension_for_pytorch==2.1.10+xpu as default
pip install --pre --upgrade ipex-llm[xpu] --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```
### 2. Configures OneAPI environment variables
```bash
source /opt/intel/oneapi/setvars.sh
```

### 3. Run

<details>

<summary>For Intel Arc™ A-Series Graphics and Intel Data Center GPU Flex Series</summary>

```bash
export USE_XETLA=OFF
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
```

</details>

<details>

<summary>For Intel Data Center GPU Max Series</summary>

```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export ENABLE_SDP_FUSION=1
```
> Note: Please note that `libtcmalloc.so` can be installed by `conda install -c conda-forge -y gperftools=2.10`.
</details>

```
python ./lookahead.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT --n-lookahead-tokens N_LOOKAHEAD
```

Arguments info:
- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Llama-2-7b-chat-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Llama-2-7b-chat-hf'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.
- `--n-lookahead-tokens`: argument defining the number of tokens to lookahead. It is 
default to be `3`.

#### Sample Output
#### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
```log
[INST] Article:```(CNN) -- A former government contract employee was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk, the Department of Justice announced Thursday. Sources say the classified materials were taken from the East Tennessee Technology Park. Roy Lynn Oakley, 67, of Roane County, Tennessee, appeared in federal court in Knoxville on Thursday. Oakley was briefly detained for questioning in the case in January, when authorities first learned of the alleged plot to divulge the materials, government sources told CNN. He voluntarily surrendered Thursday at an FBI field office in Knoxville, the sources said. Oakley is a former employee of Bechtel Jacobs, the Department of Energy's prime environmental management contractor at the East Tennessee Technology Park, prosecutors said. The indictment states that Oakley, "having possession of, access to and having been entrusted with sections of 'barriers' and associated hardware used for uranium enrichment through the process of gaseous diffusion ... having reason to believe that such data would be utilized to injure the United States and secure an advantage to a foreign nation, did communicate, transmit and disclose such data to another person." The transfer took place January 26, the indictment alleges. Oakley is also charged with converting the material and "restricted data" to his own use. He began doing so on about October 17, 2006, and continued through January, prosecutors said. Prosecutors said the materials involved have been examined by scientists and posed no threat to people who may have come into contact with them. Oakley's attorney, Herb Moncier, said outside court Thursday that Oakley's job was to break rods "into little pieces" and throw them away. Moncier said Oakley had a security clearance, but Moncier did not believe it was a high-level clearance. The government alleges that in January, Oakley attempted to sell the "pieces of scrap" to someone he thought was a French agent -- but in reality was an undercover FBI agent, Moncier said. He said he questions whether those broken pieces would be considered an "appliance" under the law. "Mr. Oakley has cooperated fully for the last six months," said Moncier, who added that he had traveled to Washington for work on the case. Each count carries a possible sentence upon conviction of up to 10 years in prison and a $250,000 fine. "While none of the stolen equipment was ever transmitted to a foreign government or terrorist organization, the facts of this case demonstrate the importance of safeguarding our nuclear technology and pursuing aggressive prosecution against those who attempt to breach the safeguards and put that technology in the wrong hands," Kenneth Wainstein, assistant attorney general for national security, said in the Justice Department statement. One government source said the materials involved are not the "crown jewels," but they should not have been taken from the facility. A "barrier" is used to filter uranium during the enrichment process, according to nuclear energy officials, but a significant number of barriers are needed to do that job. Sources told CNN that federal authorities have been following Oakley and investigating the case for at least six months, after he allegedly tried to sell the classified material. Oakley, described as a low-level employee, apparently did not make contact with any foreign government and is not a foreign agent of any kind, an official familiar with the case said. A government official with with knowledge of the case said that when authorities learned of Oakley's alleged intentions six months ago, the FBI and Department of Energy launched a joint investigation. The FBI then developed a sting operation, government officials familiar with the case said, and authorities intervened before there could be any involvement of a foreign country. East Tennessee Technology Park is an area of the DOE's Oak Ridge reservation "where we are currently decontaminating and decommissioning buildings that were last used in 1985," Gerald Boyd, manager of the DOE's Oak Ridge site office, said Thursday. "When they were in use, now over 20 years ago, some of the buildings at ETTP housed facilities used for the enrichment of uranium." Boyd said the technology park and the reservation "are protected by multiple layers of security systems and detection programs, both visible and unseen, meant to identify rogue employees attempting to abuse their access and position." In this case, a review of security procedures showed that the system worked and "successfully identified the individual in question," he said. E-mail to a friend . CNN's Terry Frieden and Kelli Arena contributed to this report.``` 

 Question: Can you please summarize this article? 

 [/INST]  Of course! Here is a summary of the article you provided:

A former government contract employee, Roy Lynn Oakley, 67, of Roane County, Tennessee, was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk. Oakley, a former employee of Bechtel Jacobs, the Department of Energy's prime environmental management contractor at the East Tennessee Technology Park, was charged with communicating, transmitting, and disclosing classified data to a foreign person. The indictment states that Oakley had reason to believe that the data
Tokens generated xxx
E2E Generation time x.xxxxs
First token latency x.xxxxs
```

#### [meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)
```log
[INST] Article:```(CNN) -- A former government contract employee was indicted on charges of stealing restricted nuclear energy-related materials and putting the United States at risk, the Department of Justice announced Thursday. Sources say the classified materials were taken from the East Tennessee Technology Park. Roy Lynn Oakley, 67, of Roane County, Tennessee, appeared in federal court in Knoxville on Thursday. Oakley was briefly detained for questioning in the case in January, when authorities first learned of the alleged plot to divulge the materials, government sources told CNN. He voluntarily surrendered Thursday at an FBI field office in Knoxville, the sources said. Oakley is a former employee of Bechtel Jacobs, the Department of Energy's prime environmental management contractor at the East Tennessee Technology Park, prosecutors said. The indictment states that Oakley, "having possession of, access to and having been entrusted with sections of 'barriers' and associated hardware used for uranium enrichment through the process of gaseous diffusion ... having reason to believe that such data would be utilized to injure the United States and secure an advantage to a foreign nation, did communicate, transmit and disclose such data to another person." The transfer took place January 26, the indictment alleges. Oakley is also charged with converting the material and "restricted data" to his own use. He began doing so on about October 17, 2006, and continued through January, prosecutors said. Prosecutors said the materials involved have been examined by scientists and posed no threat to people who may have come into contact with them. Oakley's attorney, Herb Moncier, said outside court Thursday that Oakley's job was to break rods "into little pieces" and throw them away. Moncier said Oakley had a security clearance, but Moncier did not believe it was a high-level clearance. The government alleges that in January, Oakley attempted to sell the "pieces of scrap" to someone he thought was a French agent -- but in reality was an undercover FBI agent, Moncier said. He said he questions whether those broken pieces would be considered an "appliance" under the law. "Mr. Oakley has cooperated fully for the last six months," said Moncier, who added that he had traveled to Washington for work on the case. Each count carries a possible sentence upon conviction of up to 10 years in prison and a $250,000 fine. "While none of the stolen equipment was ever transmitted to a foreign government or terrorist organization, the facts of this case demonstrate the importance of safeguarding our nuclear technology and pursuing aggressive prosecution against those who attempt to breach the safeguards and put that technology in the wrong hands," Kenneth Wainstein, assistant attorney general for national security, said in the Justice Department statement. One government source said the materials involved are not the "crown jewels," but they should not have been taken from the facility. A "barrier" is used to filter uranium during the enrichment process, according to nuclear energy officials, but a significant number of barriers are needed to do that job. Sources told CNN that federal authorities have been following Oakley and investigating the case for at least six months, after he allegedly tried to sell the classified material. Oakley, described as a low-level employee, apparently did not make contact with any foreign government and is not a foreign agent of any kind, an official familiar with the case said. A government official with with knowledge of the case said that when authorities learned of Oakley's alleged intentions six months ago, the FBI and Department of Energy launched a joint investigation. The FBI then developed a sting operation, government officials familiar with the case said, and authorities intervened before there could be any involvement of a foreign country. East Tennessee Technology Park is an area of the DOE's Oak Ridge reservation "where we are currently decontaminating and decommissioning buildings that were last used in 1985," Gerald Boyd, manager of the DOE's Oak Ridge site office, said Thursday. "When they were in use, now over 20 years ago, some of the buildings at ETTP housed facilities used for the enrichment of uranium." Boyd said the technology park and the reservation "are protected by multiple layers of security systems and detection programs, both visible and unseen, meant to identify rogue employees attempting to abuse their access and position." In this case, a review of security procedures showed that the system worked and "successfully identified the individual in question," he said. E-mail to a friend . CNN's Terry Frieden and Kelli Arena contributed to this report.``` 

 Question: Can you please summarize this article? 

 [/INST]  Sure! Here's a summary of the article:

A former government contract employee, Roy Lynn Oakley, was indicted on charges of stealing restricted nuclear energy-related materials from the East Tennessee Technology Park. He is accused of communicating, transmitting, and disclosing the materials to another person, and of converting the materials and restricted data to his own use. The materials involved have been examined by scientists and posed no threat to people who may have come into contact with them. Oakley attempted to sell the materials to someone he thought was a French agent, but it was actually an under
Tokens generated xxx
E2E Generation time x.xxxs
First token latency x.xxxs
```
