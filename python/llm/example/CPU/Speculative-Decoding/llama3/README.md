# LLaMA3

In this directory, you will find examples on how you could run LLaMA3 BF16 inference with self-speculative decoding using IPEX-LLM on [Intel CPUs](../README.md). For illustration purposes, we utilize the [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) as reference Llama3 models.

## 0. Requirements

To run these examples with IPEX-LLM on Intel CPUs, we have some recommended requirements for your machine, please refer to [here](../README.md#recommended-requirements) for more information.

## Example: Predict Tokens using `generate()` API

In the example [speculative.py](./speculative.py), we show a basic use case for a Llama2 model to predict the next N tokens using `generate()` API, with IPEX-LLM speculative decoding optimizations on Intel CPUs.

### 1. Install

We suggest using conda to manage environment:

```bash
conda create -n llm python=3.11
conda activate llm
pip install --pre --upgrade ipex-llm[all]
pip install intel_extension_for_pytorch==2.1.0
```

### 2. Configures high-performing processor environment variables

```bash
source ipex-llm-init -t
export OMP_NUM_THREADS=48 # you can change 48 here to #cores of one processor socket
```

### 3. Run

We recommend to use `numactl` to bind the program to a specified processor socket:

```bash
numactl -C 0-47 -m 0 python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```

For example, 0-47 means bind the python program to core list 0-47 for a 48-core socket.

Arguments info:

- `--repo-id-or-model-path REPO_ID_OR_MODEL_PATH`: argument defining the huggingface repo id for the Llama2 model (e.g. `meta-llama/Meta-Llama-3-8B-Instruct`) to be downloaded, or the path to the huggingface checkpoint folder. It is default to be `'meta-llama/Meta-Llama-3-8B-Instruct'`.
- `--prompt PROMPT`: argument defining the prompt to be infered (with integrated prompt format for chat). A default prompt is provided.
- `--n-predict N_PREDICT`: argument defining the max number of tokens to predict. It is default to be `128`.

#### Sample Output

#### [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

```log
E2E Generation time xx.xxxxs
user
In the year 2048, the world was a very different place from what it had been just two decades before. The pace of technological progress had quickened to an almost unimaginable degree, and the changes that had swept through society as a result were nothing short of revolutionary.
In many ways, the year 2048 represented the culmination of a long and tumultuous journey that humanity had been on since the dawn of civilization. The great leaps forward in science and technology that had occurred over the course of the previous century had laid the groundwork for a future that was beyond anything anyone could have imagined.
One of the most striking aspects of life in 2048 was the degree to which technology had become an integral part of nearly every aspect of daily existence. From the moment people woke up in the morning until they went to bed at night, they were surrounded by devices and systems that were powered by advanced artificial intelligence and machine learning algorithms.
In fact, it was hard to find anything in people's lives that wasn't touched by technology in some way. Every aspect of society had been transformed, from the way people communicated with one another to the way they worked, played, and even socialized. And as the years went on, it seemed as though there was no limit to what technology could achieve.
Despite all of these advances, however, not everyone was happy with the state of the world in 2048. Some people saw the increasing reliance on technology as a sign that humanity was losing touch with its own humanity, and they worried about the implications of this for the future.
Others were more pragmatic, recognizing that while technology had brought many benefits, it also posed new challenges and risks that needed to be addressed. As a result, there was a growing movement of people who were working to ensure that the advances of technology were used in ways that were safe, ethical, and beneficial for everyone.
One person who was at the forefront of this movement was a young woman named Maya. Maya was a brilliant and ambitious researcher who had dedicated her life to understanding the implications of emerging technologies like artificial intelligence and biotechnology. She was deeply concerned about the potential risks and unintended consequences of these technologies, and she worked tirelessly to raise awareness about the need for responsible innovation.
Maya's work had earned her a reputation as one of the most influential voices in the field of technology and ethics, and she was widely respected for her deep understanding of the issues and her ability to communicate complex ideas in ways that were accessible and engaging. She was also known for her passionate and inspiring speeches, which often left her audiences with a sense of purpose and determination to make the world a better place through their own efforts.
One day, Maya received an invitation to speak at a major conference on technology and ethics, which was being held in a large convention center in the heart of the city. The conference was expected to attract thousands of people from all over the world, and there was a great deal of excitement and anticipation about what Maya would say.
As she prepared for her speech, Maya knew that she had a big responsibility on her shoulders. She felt a deep sense of obligation to use her platform to inspire others to take action and make a difference in the world, and she was determined to do everything in her power to live up to this responsibility.
When the day of the conference arrived, Maya was filled with a mixture of excitement and nerves. She spent hours rehearsing her speech and fine-tuning her ideas, making sure that she had everything just right. Finally, after what felt like an eternity, it was time for her to take the stage.
As she stepped up to the podium, Maya could feel the energy of the crowd surging around her. She took a deep breath and began to speak, her voice strong and clear as she outlined the challenges and opportunities facing society in the age of technology. She spoke passionately about the need for responsible innovation and the importance of considering the ethical implications of our actions, and she inspired many people in the audience to take up this cause and make a difference in their own lives.
Overall, Maya's speech was a resounding success, and she received countless messages of gratitude and appreciation from those who had heard her speak. She knew that there was still much work to be done, but she felt hopeful about the future and the role that technology could play in creating a better world for all.
As Maya left the stage and made her way back to her seat, she couldn't help but feel a sense of pride and accomplishment at what she had just accomplished. She knew that her words had the power to inspire others and make a real difference in the world, and she was grateful for the opportunity to have played a part in this important work.
For Maya, the future was full of promise and possibility, and she was determined to continue doing everything in her power to help create a brighter, more ethical world for everyone.
As sheassistant
What a wonderful story! It's great to see a character like Maya, who is passionate about using technology for good and is making a positive impact on the world
Tokens generated 32
First token latency xx.xxxxs
```

### 4. Accelerate with BIGDL_OPT_IPEX

To accelerate speculative decoding on CPU, you can install our validated version of [IPEX 2.2.0+cpu](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu) refering to [IPEX's installation guide](https://intel.github.io/intel-extension-for-pytorch/index.html#installation?platform=cpu&version=v2.2.0%2Bcpu), or by the following commands: (Other versions of IPEX may have some conflicts and can not accelerate speculative decoding correctly.)

```bash
# Install IPEX 2.2.0+cpu
python -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch==2.2.0
python -m pip install oneccl_bind_pt==2.2.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
# if there is any installation problem for oneccl_binding, you can also find suitable index url at "https://pytorch-extension.intel.com/release-whl/stable/cpu/cn/" or "https://developer.intel.com/ipex-whl-stable-cpu" according to your environment.

# Update transformers
pip install transformers==4.36.2
```

After installed IPEX, you can set `BIGDL_OPT_IPEX=true` to get target model acceleration. Currently `Llama-3-8B-Instruct` are supported.

```bash
source ipex-llm-init -t
export BIGDL_OPT_IPEX=true
export OMP_NUM_THREADS=48 # you can change 48 here to #cores of one processor socket
numactl -C 0-47 -m 0 python ./speculative.py --repo-id-or-model-path REPO_ID_OR_MODEL_PATH --prompt PROMPT --n-predict N_PREDICT
```
