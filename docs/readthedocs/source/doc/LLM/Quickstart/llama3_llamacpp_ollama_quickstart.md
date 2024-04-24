# Run Llama 3 on Intel GPU using llama.cpp and ollama with IPEX-LLM

[Llama 3](https://llama.meta.com/llama3/) is the latest Large Language Models released by [Meta](https://llama.meta.com/) which provides state-of-the-art performance and excels at language nuances, contextual understanding, and complex tasks like translation and dialogue generation.

Now, you can easily run Llama 3 on Intel GPU using `llama.cpp` and `Ollama` with IPEX-LLM.

See the demo of running Llama-3-8B-Instruct on Intel Arc GPU using `Ollama` below.

<video src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-linux-arc.mp4" width="100%" controls></video>

## Quick Start
This quickstart guide walks you through how to run Llama 3 on Intel GPU using `llama.cpp` / `Ollama` with IPEX-LLM.

### 1. Run Llama 3 using llama.cpp

#### 1.1 Install IPEX-LLM for llama.cpp and Initialize

Visit [Run llama.cpp with IPEX-LLM on Intel GPU Guide](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html), and follow the instructions in section [Prerequisites](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#prerequisites) to setup and section [Install IPEX-LLM for llama.cpp](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#install-ipex-llm-for-llama-cpp) to install the IPEX-LLM with llama.cpp binaries, then follow the instructions in section [Initialize llama.cpp with IPEX-LLM](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#initialize-llama-cpp-with-ipex-llm) to initialize.

**After above steps, you should have created a conda environment, named `llm-cpp` for instance and have llama.cpp binaries in your current directory.**

**Now you can use these executable files by standard llama.cpp usage.**

#### 1.2 Download Llama3

There already are some GGUF models of Llama3 in community, here we take [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) for example.

Suppose you have downloaded a [Meta-Llama-3-8B-Instruct-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf) model from [Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF) and put it under `<model_dir>`.

#### 1.3 Run Llama3 on Intel GPU using llama.cpp

##### Set Environment Variables

Configure oneAPI variables by running the following command:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         source /opt/intel/oneapi/setvars.sh

   .. tab:: Windows

      .. note::

      This is a required step for APT or offline installed oneAPI. Skip this step for PIP-installed oneAPI.

      .. code-block:: bash

         call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

```

##### Run llama3

Under your current directory, exceuting below command to do inference with Llama3:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         ./main -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun doing something" -t 8 -e -ngl 33 --color --no-mmap

   .. tab:: Windows

      Please run the following command in Anaconda Prompt.

      .. code-block:: bash

        main -ngl 33 -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf -n 32 --prompt "Once upon a time, there existed a little girl who liked to have adventures. She wanted to go to places and meet new people, and have fun doing something" -e -ngl 33 --color --no-mmap
```

Under your current directory, you can also execute below command to have interactive chat with Llama3:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         ./main -ngl 33 -c 0 --interactive-first --color -e --in-prefix '<|start_header_id|>user<|end_header_id|>\n\n' --in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' -r '<|eot_id|>' -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf

   .. tab:: Windows

      Please run the following command in Anaconda Prompt.

      .. code-block:: bash

        main -ngl 33 -c 0 --interactive-first --color -e --in-prefix '<|start_header_id|>user<|end_header_id|>\n\n' --in-suffix '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' -r '<|eot_id|>' -m <model_dir>/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

Below is a sample output on Intel Arc GPU:
<img src="https://llm-assets.readthedocs.io/en/latest/_images/llama3-cpp-arc-demo.png" width=100%; />

### 2. Run Llama3 using Ollama

#### 2.1 Install IPEX-LLM for Ollama and Initialize

Visit [Run Ollama with IPEX-LLM on Intel GPU](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html), and follow the instructions in section [Install IPEX-LLM for llama.cpp](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/llama_cpp_quickstart.html#install-ipex-llm-for-llama-cpp) to install the IPEX-LLM with Ollama binary, then follow the instructions in section [Initialize Ollama](https://ipex-llm.readthedocs.io/en/latest/doc/LLM/Quickstart/ollama_quickstart.html#initialize-ollama) to initialize.

**After above steps, you should have created a conda environment, named `llm-cpp` for instance and have ollama binary file in your current directory.**

**Now you can use this executable file by standard Ollama usage.**

#### 2.2 Run Llama3 on Intel GPU using Ollama

[ollama/ollama](https://github.com/ollama/ollama) has alreadly added [Llama3](https://ollama.com/library/llama3) into its library, so it's really easy to run Llama3 using ollama now.

##### 2.2.1 Run Ollama Serve

Launch the Ollama service:

```eval_rst
.. tabs::
   .. tab:: Linux

      .. code-block:: bash

         export no_proxy=localhost,127.0.0.1
         export ZES_ENABLE_SYSMAN=1
         export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
         export OLLAMA_NUM_GPU=999
         source /opt/intel/oneapi/setvars.sh

         ./ollama serve

   .. tab:: Windows

      Please run the following command in Anaconda Prompt.

      .. code-block:: bash

         set no_proxy=localhost,127.0.0.1
         set ZES_ENABLE_SYSMAN=1
         set OLLAMA_NUM_GPU=999
         # Below is a required step for APT or offline installed oneAPI. Skip below step for PIP-installed oneAPI.
         call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

         ollama serve

```

```eval_rst
.. note::

  To allow the service to accept connections from all IP addresses, use `OLLAMA_HOST=0.0.0.0 ./ollama serve` instead of just `./ollama serve`.
```

##### 2.2.2 Using Ollama Run Llama3

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

```eval_rst
.. note::

  Here we just take `llama3:8b-instruct-q4_K_M` for example, you can replace it with any other Llama3 model you want.
```

Below is a sample output on Intel Arc GPU :
<img src="https://llm-assets.readthedocs.io/en/latest/_images/ollama-llama3-arc-demo.png" width=100%; />
