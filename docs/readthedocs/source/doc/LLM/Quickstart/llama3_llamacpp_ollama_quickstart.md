#  Run Llama 3 on Intel GPU using llama.cpp and ollama with IPEX-LLM

[ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) prvoides fast LLM inference in in pure C++ across a variety of hardware, [ollama/ollama](https://github.com/ollama/ollama) is popular framework designed to build and run language models on a local machine. You can now use the C++ interface of [`ipex-llm`](https://github.com/intel-analytics/ipex-llm) as an accelerated backend for `llama.cpp` and `ollama` running on Intel **GPU** *(e.g., local PC with iGPU, discrete GPU such as Arc, Flex and Max)*.

[Llama 3](https://llama.meta.com/llama3/) is the latest Large Language Models released by [Meta](https://llama.meta.com/) which provides state-of-the-art performance and excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation. 

Now, you can easily run Llama 3 GGUF models on Intel GPU using `llama.cpp` and `Ollama` with IPEX-LLM.

See the demo of running Llama-3-8B-Instruct-Q4_K_M.gguf on Intel Arc GPU below.

[ollama linux arc demo to be recorded]

## Quick Start
This quickstart guide walks you through how to run Llama 3 on Intel GPU using `llama.cpp` / `Ollama` with IPEX-LLM.

### 1 Install IPEX-LLM for llama.cpp and Ollama

Visit [Run llama.cpp with IPEX-LLM on Intel GPU Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html), and follow the instructions in section [Prerequisites](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#prerequisites) to setup and section [Install IPEX-LLM cpp](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#install-ipex-llm-for-llama-cpp) to install the IPEX-LLM with llama.cpp and Ollama binaries. 

**After the installation, you should have created a conda environment, named `llm-cpp` for instance.**

### 2. Initialize llama.cpp and Ollama

Activate the `llm-cpp` conda environment and initialize llama.cpp and Ollama by executing the commands below. Many symbolic link of `llama.cpp` and `Ollama` will appear in your current directory.

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         conda activate llm-cpp
         init-llama-cpp
         init-ollama

   .. tab:: Windows

      Please run the following command with **administrator privilege in Anaconda Prompt**.

      .. code-block:: bash

         conda activate llm-cpp
         init-llama-cpp.bat
         init-ollama.bat

```

**Now you can use these executable file by standard llama.cpp or ollama's usage.**

### 3. Run Llama3 on Intel GPU using llama.cpp

There already are some GGUF models of Llama3 in community, here we take [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) for example.

Suppose you have downloaded a [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf] model from [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) and put it under <model_dir>.

#### 3.1 Continuation


#### 3.2 Interative chat


### 4. Run Llama3 on Intel GPU using ollama

[ollama/ollama](https://github.com/ollama/ollama) has alreadly added [Llama3](https://ollama.com/library/llama3) into its library, so it's really easy to run Llama3 using ollama now.

#### 4.1 Run Ollama Serve

Launch the Ollama service:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         export ZES_ENABLE_SYSMAN=1
         export OLLAMA_NUM_GPU=999
         source /opt/intel/oneapi/setvars.sh

         ./ollama serve

   .. tab:: Windows

      Please run the following command in Anaconda Prompt.

      .. code-block:: bash

         set no_proxy=localhost,127.0.0.1
         set ZES_ENABLE_SYSMAN=1
         set OLLAMA_NUM_GPU=999
         call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

         ollama serve

```

```eval_rst
.. note::

  To allow the service to accept connections from all IP addresses, use `OLLAMA_HOST=0.0.0.0 ./ollama serve` instead of just `./ollama serve`.
```

#### 4.2 Using Ollama Run Llama3

Keep the Ollama service on and open another terminal and run llama3 with `ollama run`:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         ./ollama run llama3:8b-instruct-q4_K_M

   .. tab:: Windows

      Please run the following command in Anaconda Prompt.

      .. code-block:: bash

         set no_proxy=localhost,127.0.0.1
         ollama run llama3:8b-instruct-q4_K_M
```

[Put a screenshot or demo on MTL here]

For more usage, you can refer to this [Run Ollama with IPEX-LLM on Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html#) QuickStart.
