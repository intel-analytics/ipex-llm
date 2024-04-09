## Running AutoGen Agent Chat with IPEX-LLM on Local Models
This example is adapted from the [Official AutoGen Teachablility tutorial](https://github.com/microsoft/autogen/blob/main/notebook/agentchat_teachability.ipynb). We use a version of FastChat modified for IPEX-LLM to create a teachable chat agent with [AutoGen](https://microsoft.github.io/autogen/) that works with locally deployed LLMs. This special agent can remember things you tell it over time, unlike regular chatbots that forget after each conversation. It does this by saving what it learns on disk, and then bring up the learnt information in future chats. This means you can teach it lots of new things—like facts, new skills, preferences, etc.

In this example, we illustrate teaching the agent something it doesn't initially know. When we ask, `What is the Vicuna model?`, it doesn't have the answer. We then inform it, `Vicuna is a 13B-parameter language model released by Meta.` We repeat the process for the Orca model, telling the agent, `Orca is a 13B-parameter language model developed by Microsoft. It outperforms Vicuna on most tasks.` Finally, we test if the agent has learned by asking, `How does the Vicuna model compare to the Orca model?` The agent's response confirms it has retained and can use the information we taught it.


### 1. Setup IPEX-LLM Environment
```bash
# create autogen running directory
mkdir autogen
cd autogen

# create respective conda environment
conda create -n autogen python=3.11
conda activate autogen

# install fastchat-adapted ipex-llm
# we recommend using ipex-llm version >= 2.5.0b20240110
pip install --pre --upgrade ipex-llm[serving]

# install recommend transformers version
pip install transformers==4.36.2

# install necessary dependencies
pip install chromadb==0.4.22
```

### 2. Setup FastChat and AutoGen Environment
```bash
# clone the FastChat in the autogen folder
git clone https://github.com/lm-sys/FastChat.git FastChat # clone the FastChat
cd FastChat
pip3 install --upgrade pip  # enable PEP 660 support
pip3 install -e ".[model_worker,webui]"

# setup AutoGen environment
pip install pyautogen==0.2.7
```

**After setting up the environment, the folder structure should be:**
```
-- autogen
| -- FastChat
```


### 3. Build FastChat OpenAI-Compatible RESTful API
Open 3 terminals

**Terminal 1: Launch the controller**

```bash
# activate conda environment
conda activate autogen

# go to the cloned FastChat folder in autogen folder
cd autogen/FastChat

python -m fastchat.serve.controller
```

**Terminal 2: Launch the workers**

```bash
# activate conda environment
conda activate autogen

# go to the created autogen folder
cd autogen

# load the local model with cpu with your downloaded model
python -m ipex_llm.serving.model_worker --model-path ... --device cpu
```

Change the Model Name:
> Assume you use the model `Mistral-7B-Instruct-v0.2` and your model is downloaded to `autogen/model/Mistral-7B-Instruct-v0.2`. You should rename the model to `autogen/model/ipex-llm` and run `python -m ipex_llm.serving.model_worker --model-path ... --device cpu`. This ensures the proper usage of the IPEX-LLM-adapted FastChat.

Potential Error Note:
> If you get `RuntimeError: Error register to Controller` in the worker terminal, please set `export no_proxy='localhost'` to ensure the registration


**Terminal 3: Launch the server**

```bash
# activate conda environment
conda activate autogen

# go to the cloned FastChat folder in autogen folder
cd autogen/FastChat

python -m fastchat.serve.openai_api_server --host localhost --port 8000
```

### 4. Run Example
Open another terminal

```bash
# activate conda environment
conda activate autogen

# go to the autogen folder
cd autogen

# run the autogen example
python teachability_new_knowledge.py
```

Potential Error Note:
> If you get `?bu=http://localhost:8000/v1/chat/completions&bc=Failed+to+retrieve+requested+URL.&ip=10.239.44.101&er=ERR_CONNECT_FAIL` in the running terminal, please set `export no_proxy='localhost'` to ensure the registration.


## Sample Output

**Using `Mistral-7B-Instruct-v0.2` model on Intel i9-12900K**

```bash
CLEARING MEMORY
user (to teachable_agent):

What is the Vicuna model?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
teachable_agent (to user):

I apologize for any confusion, but I don't have enough context or prior information from our conversations to know specifically what you mean by "the Vicuna model." Vicunas are a species of camelid native to South America, but there is no known statistical or machine learning model named after them in the field of data science or artificial intelligence. If you could please provide more context or details about what you mean by "the Vicuna model," I would be happy to help you with any related questions or information you might have.

--------------------------------------------------------------------------------
user (to teachable_agent):

Vicuna is a 13B-parameter language model released by Meta.

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
teachable_agent (to user):

Thank you for providing the additional context. Based on the new information, the Vicuna model is a 13B-parameter language model developed and released by Meta (formerly Facebook). However, I would need to access more detailed information about this model to fully understand its capabilities, architecture, and applications. If you have any specific questions or details about the Vicuna model, feel free to ask and I will do my best to help you.

--------------------------------------------------------------------------------
user (to teachable_agent):

What is the Orca model?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
teachable_agent (to user):

I apologize for any confusion, but I cannot directly answer your question about the Orca model without more context or information. The term "Orca model" is not mentioned in the context provided in our previous conversation, which was about the Vicuna model being a 13B-parameter language model released by Meta. If you could please provide more details or context about the Orca model, such as who developed it, when it was released, and what type of model it is, I would be happy to help you with any related questions or information you might have.

--------------------------------------------------------------------------------
user (to teachable_agent):

Orca is a 13B-parameter language model developed by Microsoft. It outperforms Vicuna on most tasks.

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
teachable_agent (to user):

Thank you for providing the additional context about the Orca model being a 13B-parameter language model developed by Microsoft that outperforms the Vicuna model on most tasks. With this information, we can infer that both the Vicuna and Orca models are large-scale language models with a similar number of parameters, and that the Orca model has better performance based on the tasks it has been evaluated on. However, without more specific details about the models' architectures, capabilities, and applications, it is difficult to provide a comprehensive comparison or analysis. If you have any specific questions or details about the Vicuna or Orca models, feel free to ask and I will do my best to help you.

--------------------------------------------------------------------------------
user (to teachable_agent):

How does the Vicuna model compare to the Orca model?

--------------------------------------------------------------------------------

>>>>>>>> USING AUTO REPLY...
teachable_agent (to user):

Based on the given memories, the Vicuna model and the Orca model are both 13B-parameter language models, meaning they have similar capacity and architecture. However, the text states that the Orca model, developed by Microsoft, outperforms the Vicuna model on most tasks. Therefore, the Orca model can be considered more advanced or effective than the Vicuna model based on the provided information. It's important to note that this comparison is based on the specific task or set of tasks mentioned in the text, and the performance of the models may vary depending on the specific use case or dataset.

--------------------------------------------------------------------------------
```
