#  💫 Intel® LLM Library for PyTorch* 
<p>
  < <a href='./README.md'>English</a> | <a href='./README.zh-CN.md'>中文</a> | <b>日本語</b> >
</p>

**`IPEX-LLM`**は、Intel ***CPU***、***GPU***（例：iGPUを搭載したローカルPC、Arc、Flex、MaxなどのディスクリートGPU）および***NPU***[^1]用のLLM加速ライブラリです。
> [!NOTE]
> - *これは、**`llama.cpp`**、**`transformers`**、**`bitsandbytes`**、**`vLLM`**、**`qlora`**、**`AutoGPTQ`**、**`AutoAWQ`**などの優れた作業に基づいて構築されています。*
> - *これは、[llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md)、[Ollama](docs/mddocs/Quickstart/ollama_quickstart.md)、[HuggingFace transformers](python/llm/example/GPU/HuggingFace)、[LangChain](python/llm/example/GPU/LangChain)、[LlamaIndex](python/llm/example/GPU/LlamaIndex)、[vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md)、[Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md)、[DeepSpeed-AutoTP](python/llm/example/GPU/Deepspeed-AutoTP)、[FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md)、[Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md)、[HuggingFace PEFT](python/llm/example/GPU/LLM-Finetuning)、[HuggingFace TRL](python/llm/example/GPU/LLM-Finetuning/DPO)、[AutoGen](python/llm/example/CPU/Applications/autogen)、[ModeScope](python/llm/example/GPU/ModelScope-Models)とシームレスに統合されています。*
> - ***70以上のモデル**が`ipex-llm`で最適化/検証されており（例：Llama、Phi、Mistral、Mixtral、Whisper、Qwen、MiniCPM、Qwen-VL、MiniCPM-Vなど）、最先端の**LLM最適化**、**XPU加速**、および**低ビット（FP8/FP6/FP4/INT4）サポート**を提供しています。完全なリストは[こちら](#verified-models)をご覧ください。*

## 最新の更新 🔥 
- [2024/07] Microsoftの**GraphRAG**をIntel GPUで実行するサポートを追加しました。クイックスタートガイドは[こちら](docs/mddocs/Quickstart/graphrag_quickstart.md)をご覧ください。
- [2024/07] 大規模マルチモーダルモデルのサポートを大幅に強化しました。詳細は[こちら](python/llm/example/GPU/HuggingFace/Multimodal)をご覧ください。
- [2024/07] Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types)で**FP6**のサポートを追加しました。
- [2024/06] Intel Core Ultraプロセッサの**NPU**サポートを実験的に追加しました。詳細は[こちら](python/llm/example/NPU/HF-Transformers-AutoModels)をご覧ください。
- [2024/06] **パイプライン並列**[推論](python/llm/example/GPU/Pipeline-Parallel-Inference)のサポートを大幅に強化しました。これにより、2つ以上のIntel GPU（例：Arc）を使用して大規模なLLMを実行することが容易になります。
- [2024/06] Intel [GPU](docs/mddocs/Quickstart/ragflow_quickstart.md)で**RAGFlow**を実行するサポートを追加しました。
- [2024/05] **Axolotl**を使用してIntel GPUでLLMの微調整を行うサポートを追加しました。クイックスタートガイドは[こちら](docs/mddocs/Quickstart/axolotl_quickstart.md)をご覧ください。

<details><summary>さらに多くの更新</summary>
<br/>
 
- [2024/05] **Docker** [images](#docker)を使用して、`ipex-llm`の推論、サービス、微調整を簡単に実行できます。
- [2024/05] Windowsで`ipex-llm`をインストールするための"*[one command](docs/mddocs/Quickstart/install_windows_gpu.md#install-ipex-llm)*"を追加しました。
- [2024/04] `ipex-llm`を使用してIntel GPUで**Open WebUI**を実行するサポートを追加しました。クイックスタートガイドは[こちら](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md)をご覧ください。
- [2024/04] `ipex-llm`を使用してIntel GPUで**Llama 3**を実行するサポートを追加しました。クイックスタートガイドは[こちら](docs/mddocs/Quickstart/llama3_llamacpp_ollama_quickstart.md)をご覧ください。
- [2024/04] `ipex-llm`はIntel [GPU](python/llm/example/GPU/HuggingFace/LLM/llama3)および[CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3)で**Llama 3**をサポートしています。
- [2024/04] `ipex-llm`はC++インターフェースを提供しており、Intel GPUで[llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md)および[ollama](docs/mddocs/Quickstart/ollama_quickstart.md)を実行するための加速バックエンドとして使用できます。
- [2024/03] `bigdl-llm`は`ipex-llm`に改名されました（移行ガイドは[こちら](docs/mddocs/Quickstart/bigdl_llm_migration.md)をご覧ください）。元の`BigDL`プロジェクトは[こちら](https://github.com/intel-analytics/bigdl-2.x)で見つけることができます。
- [2024/02] `ipex-llm`は[ModelScope](python/llm/example/GPU/ModelScope-Models)（[魔搭](python/llm/example/CPU/ModelScope-Models)）から直接モデルをロードするサポートを追加しました。
- [2024/02] `ipex-llm`は初期の**INT2**サポートを追加しました（llama.cpp [IQ2](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2)メカニズムに基づく）。これにより、16GBのVRAMを持つIntel GPUで大規模なLLM（例：Mixtral-8x7B）を実行することが可能になります。
- [2024/02] ユーザーは[Text-Generation-WebUI](https://github.com/intel-analytics/text-generation-webui) GUIを通じて`ipex-llm`を使用できます。
- [2024/02] `ipex-llm`は*[Self-Speculative Decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md)*をサポートしており、Intel [GPU](python/llm/example/GPU/Speculative-Decoding)および[CPU](python/llm/example/CPU/Speculative-Decoding)でのFP16およびBF16推論のレイテンシを**約30％**短縮します。
- [2024/02] `ipex-llm`はIntel GPUでの包括的なLLM微調整をサポートしています（[LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA)、[QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA)、[DPO](python/llm/example/GPU/LLM-Finetuning/DPO)、[QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA)、[ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)を含む）。
- [2024/01] `ipex-llm` [QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA)を使用して、8つのIntel Max 1550 GPUで[Standford-Alpaca](python/llm/example/GPU/LLM-Finetuning/QLoRA/alpaca-qlora)を使用してLLaMA2-7Bを**21分**、LLaMA2-70Bを**3.14時間**で微調整しました（ブログは[こちら](https://www.intel.com/content/www/us/en/developer/articles/technical/finetuning-llms-on-intel-gpus-using-bigdl-llm.html)をご覧ください）。
- [2023/12] `ipex-llm`は[ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)をサポートしています（詳細は*["ReLoRA: High-Rank Training Through Low-Rank Updates"](https://arxiv.org/abs/2307.05695)*をご覧ください）。
- [2023/12] `ipex-llm`はIntel [GPU](python/llm/example/GPU/HuggingFace/LLM/mixtral)および[CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral)で[Mixtral-8x7B](python/llm/example/GPU/HuggingFace/LLM/mixtral)をサポートしています。
- [2023/12] `ipex-llm`は[QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA)をサポートしています（詳細は*["QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2309.14717)*をご覧ください）。
- [2023/12] `ipex-llm`はIntel ***GPU***での[FP8およびFP4推論](python/llm/example/GPU/HuggingFace/More-Data-Types)をサポートしています。
- [2023/11] 初期のサポートとして、[GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF)、[AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ)、および[GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ)モデルを直接`ipex-llm`にロードすることができます。
- [2023/11] `ipex-llm`はIntel [GPU](python/llm/example/GPU/vLLM-Serving)および[CPU](python/llm/example/CPU/vLLM-Serving)での[vLLM連続バッチ処理](python/llm/example/GPU/vLLM-Serving)をサポートしています。
- [2023/10] `ipex-llm`はIntel [GPU](python/llm/example/GPU/LLM-Finetuning/QLoRA)および[CPU](python/llm/example/CPU/QLoRA-FineTuning)での[QLoRA微調整](python/llm/example/GPU/LLM-Finetuning/QLoRA)をサポートしています。
- [2023/10] `ipex-llm`はIntel GPUおよびCPUでの[FastChatサービス](python/llm/src/ipex_llm/llm/serving)をサポートしています。
- [2023/09] `ipex-llm`は[Intel GPU](python/llm/example/GPU)（iGPU、Arc、Flex、MAXを含む）をサポートしています。
- [2023/09] `ipex-llm` [チュートリアル](https://github.com/intel-analytics/ipex-llm-tutorial)が公開されました。
 
</details> 

## `ipex-llm` パフォーマンス
以下に、Intel Core UltraおよびIntel Arc GPUでの**トークン生成速度**を示します[^1]（詳細は[[2]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-meta-llama3-with-intel-ai-solutions.html)[[3]](https://www.intel.com/content/www/us/en/developer/articles/technical/accelerate-microsoft-phi-3-models-intel-ai-soln.html)[[4]](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-ai-solutions-accelerate-alibaba-qwen2-llms.html)をご覧ください）。

<table width="100%">
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/MTL_perf.jpg" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/MTL_perf.jpg" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/Arc_perf.jpg" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/Arc_perf.jpg" width=100%; />
      </a>
    </td>
  </tr>
</table>

`ipex-llm`のパフォーマンスベンチマークを実行するには、[ベンチマークガイド](docs/mddocs/Quickstart/benchmark_quickstart.md)を参照してください。

## `ipex-llm` デモ

以下は、Intel Iris iGPU、Intel Core Ultra iGPU、単一カードArc GPU、または複数カードArc GPUを使用して`ipex-llm`でローカルLLMを実行するデモです。

<table width="100%">
  <tr>
    <td align="center" colspan="1"><strong>Intel Iris iGPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Core Ultra iGPU</strong></td>
    <td align="center" colspan="1"><strong>Intel Arc dGPU</strong></td>
    <td align="center" colspan="1"><strong>2-Card Intel Arc dGPUs</strong></td>
  </tr>
  <tr>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/iris_phi3-3.8B_q4_0_llamacpp_long.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/iris_phi3-3.8B_q4_0_llamacpp_long.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/mtl_mistral-7B_q4_k_m_ollama.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/arc_llama3-8B_fp8_textwebui.gif" width=100%; />
      </a>
    </td>
    <td>
      <a href="https://llm-assets.readthedocs.io/en/latest/_images/2arc_qwen1.5-32B_fp6_fastchat.gif" target="_blank">
        <img src="https://llm-assets.readthedocs.io/en/latest/_images/2arc_qwen1.5-32B_fp6_fastchat.gif" width=100%; />
      </a>
    </td>
  </tr>
  <tr>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/llama_cpp_quickstart.md">llama.cpp (Phi-3-mini Q4_0)</a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/ollama_quickstart.md">Ollama (Mistral-7B Q4_K) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/webui_quickstart.md">TextGeneration-WebUI (Llama3-8B FP8) </a>
    </td>
    <td align="center" width="25%">
      <a href="docs/mddocs/Quickstart/fastchat_quickstart.md">FastChat (QWen1.5-32B FP6)</a>
    </td>  </tr>
</table>

## モデルの精度
以下に、**Perplexity**の結果を示します（Wikitextデータセットを使用して[こちら](https://github.com/intel-analytics/ipex-llm/tree/main/python/llm/dev/benchmark/perplexity)のスクリプトを使用してテストしました）。
|Perplexity                 |sym_int4	|q4_k	  |fp6	  |fp8_e5m2 |fp8_e4m3 |fp16   |
|---------------------------|---------|-------|-------|---------|---------|-------|
|Llama-2-7B-chat-hf	        |6.364 	  |6.218 	|6.092 	|6.180 	  |6.098    |6.096  | 
|Mistral-7B-Instruct-v0.2	  |5.365 	  |5.320 	|5.270 	|5.273 	  |5.246	   |5.244  |
|Baichuan2-7B-chat	         |6.734    |6.727	 |6.527	 |6.539	   |6.488	   |6.508  |
|Qwen1.5-7B-chat	           |8.865 	  |8.816 	|8.557 	|8.846 	  |8.530    |8.607  | 
|Llama-3.1-8B-Instruct	     |6.705	   |6.566	 |6.338	 |6.383	   |6.325	   |6.267  |
|gemma-2-9b-it	             |7.541	   |7.412	 |7.269	 |7.380	   |7.268	   |7.270  |
|Baichuan2-13B-Chat	        |6.313	   |6.160	 |6.070	 |6.145	   |6.086	   |6.031  |
|Llama-2-13b-chat-hf	       |5.449	   |5.422	 |5.341	 |5.384	   |5.332	   |5.329  |
|Qwen1.5-14B-Chat	          |7.529	   |7.520	 |7.367	 |7.504	   |7.297	   |7.334  |

[^1]: パフォーマンスは使用方法、構成、およびその他の要因によって異なります。`ipex-llm`は、非Intel製品に対して同じ程度の最適化を行わない場合があります。詳細はwww.Intel.com/PerformanceIndexをご覧ください。

## `ipex-llm` クイックスタート

### Docker
- [GPU Inference in C++](docs/mddocs/DockerGuides/docker_cpp_xpu_quickstart.md): Intel GPUで`ipex-llm`を使用して`llama.cpp`、`ollama`などを実行する
- [GPU Inference in Python](docs/mddocs/DockerGuides/docker_pytorch_inference_gpu.md): Intel GPUで`ipex-llm`を使用してHuggingFace `transformers`、`LangChain`、`LlamaIndex`、`ModelScope`などを実行する
- [vLLM on GPU](docs/mddocs/DockerGuides/vllm_docker_quickstart.md): Intel GPUで`ipex-llm`を使用して`vLLM`推論サービスを実行する
- [vLLM on CPU](docs/mddocs/DockerGuides/vllm_cpu_docker_quickstart.md): Intel CPUで`ipex-llm`を使用して`vLLM`推論サービスを実行する
- [FastChat on GPU](docs/mddocs/DockerGuides/fastchat_docker_quickstart.md): Intel GPUで`ipex-llm`を使用して`FastChat`推論サービスを実行する
- [VSCode on GPU](docs/mddocs/DockerGuides/docker_run_pytorch_inference_in_vscode.md): Intel GPUでVSCodeを使用してPythonベースの`ipex-llm`アプリケーションを開発および実行する

### 使用
- [llama.cpp](docs/mddocs/Quickstart/llama_cpp_quickstart.md): Intel GPUで**llama.cpp**を実行する（`ipex-llm`のC++インターフェースを使用）
- [Ollama](docs/mddocs/Quickstart/ollama_quickstart.md): Intel GPUで**ollama**を実行する（`ipex-llm`のC++インターフェースを使用）
- [PyTorch/HuggingFace](docs/mddocs/Quickstart/install_windows_gpu.md): Intel GPUで**PyTorch**、**HuggingFace**、**LangChain**、**LlamaIndex**などを実行する（`ipex-llm`のPythonインターフェースを使用）[Windows](docs/mddocs/Quickstart/install_windows_gpu.md)および[Linux](docs/mddocs/Quickstart/install_linux_gpu.md)
- [vLLM](docs/mddocs/Quickstart/vLLM_quickstart.md): Intel [GPU](docs/mddocs/DockerGuides/vllm_docker_quickstart.md)および[CPU](docs/mddocs/DockerGuides/vllm_cpu_docker_quickstart.md)で`ipex-llm`を使用して**vLLM**を実行する
- [FastChat](docs/mddocs/Quickstart/fastchat_quickstart.md): Intel GPUおよびCPUで`ipex-llm`を使用して**FastChat**サービスを実行する
- [Serving on multiple Intel GPUs](docs/mddocs/Quickstart/deepspeed_autotp_fastapi_quickstart.md): DeepSpeed AutoTPおよびFastAPIを活用して複数のIntel GPUで`ipex-llm`**推論サービス**を実行する
- [Text-Generation-WebUI](docs/mddocs/Quickstart/webui_quickstart.md): `ipex-llm`を使用して`oobabooga`**WebUI**を実行する
- [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md): **Axolotl**で`ipex-llm`を使用してLLMを微調整する
- [Benchmarking](docs/mddocs/Quickstart/benchmark_quickstart.md): Intel GPUおよびCPUで`ipex-llm`の**ベンチマーク**（レイテンシおよびスループット）を実行する

### アプリケーション
- [GraphRAG](docs/mddocs/Quickstart/graphrag_quickstart.md): `ipex-llm`を使用してMicrosoftの`GraphRAG`をローカルLLMで実行する
- [RAGFlow](docs/mddocs/Quickstart/ragflow_quickstart.md): `ipex-llm`を使用して`RAGFlow`（*オープンソースのRAGエンジン*）を実行する
- [LangChain-Chatchat](docs/mddocs/Quickstart/chatchat_quickstart.md): `ipex-llm`を使用して`LangChain-Chatchat`（*RAGパイプラインを使用したナレッジベースQA*）を実行する
- [Coding copilot](docs/mddocs/Quickstart/continue_quickstart.md): `ipex-llm`を使用して`Continue`（VSCodeのコーディングコパイロット）を実行する
- [Open WebUI](docs/mddocs/Quickstart/open_webui_with_ollama_quickstart.md): `ipex-llm`を使用して`Open WebUI`を実行する
- [PrivateGPT](docs/mddocs/Quickstart/privateGPT_quickstart.md): `ipex-llm`を使用して`PrivateGPT`を実行し、ドキュメントと対話する
- [Dify platform](docs/mddocs/Quickstart/dify_quickstart.md): `Dify`（*プロダクション対応のLLMアプリ開発プラットフォーム*）で`ipex-llm`を使用する

### インストール
- [Windows GPU](docs/mddocs/Quickstart/install_windows_gpu.md): Intel GPUを搭載したWindowsで`ipex-llm`をインストールする
- [Linux GPU](docs/mddocs/Quickstart/install_linux_gpu.md): Intel GPUを搭載したLinuxで`ipex-llm`をインストールする
- *詳細については、[完全なインストールガイド](docs/mddocs/Overview/install.md)を参照してください*

### コード例
- #### 低ビット推論
  - [INT4推論](python/llm/example/GPU/HuggingFace/LLM): Intel [GPU](python/llm/example/GPU/HuggingFace/LLM)および[CPU](python/llm/example/CPU/HF-Transformers-AutoModels/Model)で**INT4** LLM推論を実行する
  - [FP8/FP6/FP4推論](python/llm/example/GPU/HuggingFace/More-Data-Types): Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types)で**FP8**、**FP6**、**FP4** LLM推論を実行する
  - [INT8推論](python/llm/example/GPU/HuggingFace/More-Data-Types): Intel [GPU](python/llm/example/GPU/HuggingFace/More-Data-Types)および[CPU](python/llm/example/CPU/HF-Transformers-AutoModels/More-Data-Types)で**INT8** LLM推論を実行する
  - [INT2推論](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2): Intel [GPU](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF-IQ2)で**INT2** LLM推論を実行する（llama.cpp IQ2メカニズムに基づく）
- #### FP16/BF16推論
  - Intel [GPU](python/llm/example/GPU/Speculative-Decoding)で**FP16** LLM推論を実行する（[self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md)最適化を使用する場合）
  - Intel [CPU](python/llm/example/CPU/Speculative-Decoding)で**BF16** LLM推論を実行する（[self-speculative decoding](docs/mddocs/Inference/Self_Speculative_Decoding.md)最適化を使用する場合）
- #### 分散推論
  - Intel [GPU](python/llm/example/GPU/Pipeline-Parallel-Inference)で**パイプライン並列**推論を実行する
  - Intel [GPU](python/llm/example/GPU/Deepspeed-AutoTP)で**DeepSpeed AutoTP**推論を実行する
- #### 保存と読み込み
  - [低ビットモデル](python/llm/example/CPU/HF-Transformers-AutoModels/Save-Load): `ipex-llm`低ビットモデル（INT4/FP4/FP6/INT8/FP8/FP16/etc.）を保存および読み込む
  - [GGUF](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GGUF): GGUFモデルを直接`ipex-llm`に読み込む
  - [AWQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/AWQ): AWQモデルを直接`ipex-llm`に読み込む
  - [GPTQ](python/llm/example/GPU/HuggingFace/Advanced-Quantizations/GPTQ): GPTQモデルを直接`ipex-llm`に読み込む
- #### 微調整
  - Intel [GPU](python/llm/example/GPU/LLM-Finetuning)でLLMを微調整する（[LoRA](python/llm/example/GPU/LLM-Finetuning/LoRA)、[QLoRA](python/llm/example/GPU/LLM-Finetuning/QLoRA)、[DPO](python/llm/example/GPU/LLM-Finetuning/DPO)、[QA-LoRA](python/llm/example/GPU/LLM-Finetuning/QA-LoRA)、[ReLoRA](python/llm/example/GPU/LLM-Finetuning/ReLora)を含む）
  - Intel [CPU](python/llm/example/CPU/QLoRA-FineTuning)でQLoRAを微調整する
- #### コミュニティライブラリとの統合
  - [HuggingFace transformers](python/llm/example/GPU/HuggingFace)
  - [Standard PyTorch model](python/llm/example/GPU/PyTorch-Models)
  - [LangChain](python/llm/example/GPU/LangChain)
  - [LlamaIndex](python/llm/example/GPU/LlamaIndex)
  - [DeepSpeed-AutoTP](python/llm/example/GPU/Deepspeed-AutoTP)
  - [Axolotl](docs/mddocs/Quickstart/axolotl_quickstart.md)
  - [HuggingFace PEFT](python/llm/example/GPU/LLM-Finetuning/HF-PEFT)
  - [HuggingFace TRL](python/llm/example/GPU/LLM-Finetuning/DPO)
  - [AutoGen](python/llm/example/CPU/Applications/autogen)
  - [ModeScope](python/llm/example/GPU/ModelScope-Models)
- [チュートリアル](https://github.com/intel-analytics/ipex-llm-tutorial)

## APIドキュメント
- [HuggingFace TransformersスタイルのAPI（Autoクラス）](docs/mddocs/PythonAPI/transformers.md)
- [任意のPyTorchモデル用のAPI](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/PythonAPI/optimize.md)

## FAQ
- [FAQとトラブルシューティング](docs/mddocs/Overview/FAQ/faq.md)

## 検証済みモデル
50以上のモデルが`ipex-llm`で最適化/検証されており、*LLaMA/LLaMA2、Mistral、Mixtral、Gemma、LLaVA、Whisper、ChatGLM2/ChatGLM3、Baichuan/Baichuan2、Qwen/Qwen-1.5、InternLM*などが含まれます。詳細は以下のリストをご覧ください。
  
| モデル      | CPU例                                                    | GPU例                                                     |
|------------|----------------------------------------------------------------|-----------------------------------------------------------------|
| LLaMA *(such as Vicuna, Guanaco, Koala, Baize, WizardLM, etc.)* | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/vicuna) |[link](python/llm/example/GPU/HuggingFace/LLM/vicuna)|
| LLaMA 2    | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama2) | [link](python/llm/example/GPU/HuggingFace/LLM/llama2)  |
| LLaMA 3    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3)  |
| LLaMA 3.1    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/llama3.1) | [link](python/llm/example/GPU/HuggingFace/LLM/llama3.1)  |
| LLaMA 3.2    |  | [link](python/llm/example/GPU/HuggingFace/LLM/llama3.2)  |
| LLaMA 3.2-Vision    |  | [link](python/llm/example/GPU/PyTorch-Models/Model/llama3.2-vision/)  |
| ChatGLM    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm)   |    | 
| ChatGLM2   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm2)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm2)   |
| ChatGLM3   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/chatglm3)  | [link](python/llm/example/GPU/HuggingFace/LLM/chatglm3)   |
| GLM-4      | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm4)      | [link](python/llm/example/GPU/HuggingFace/LLM/glm4)       |
| GLM-4V     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/glm-4v)    | [link](python/llm/example/GPU/HuggingFace/Multimodal/glm-4v)     |
| Mistral    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mistral)   | [link](python/llm/example/GPU/HuggingFace/LLM/mistral)    |
| Mixtral    | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mixtral)   | [link](python/llm/example/GPU/HuggingFace/LLM/mixtral)    |
| Falcon     | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/falcon)    | [link](python/llm/example/GPU/HuggingFace/LLM/falcon)     |
| MPT        | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/mpt)       | [link](python/llm/example/GPU/HuggingFace/LLM/mpt)        |
| Dolly-v1   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v1)  | [link](python/llm/example/GPU/HuggingFace/LLM/dolly-v1)   | 
| Dolly-v2   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/dolly_v2)  | [link](python/llm/example/GPU/HuggingFace/LLM/dolly-v2)   | 
| Replit Code| [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/replit)    | [link](python/llm/example/GPU/HuggingFace/LLM/replit)     |
| RedPajama  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/redpajama) |    | 
| Phoenix    | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/phoenix)   |    | 
| StarCoder  | [link1](python/llm/example/CPU/Native-Models), [link2](python/llm/example/CPU/HF-Transformers-AutoModels/Model/starcoder) | [link](python/llm/example/GPU/HuggingFace/LLM/starcoder) | 
| Baichuan   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan)  | [link](python/llm/example/GPU/HuggingFace/LLM/baichuan)   |
| Baichuan2  | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/baichuan2) | [link](python/llm/example/GPU/HuggingFace/LLM/baichuan2)  |
| InternLM   | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/internlm)  | [link](python/llm/example/GPU/HuggingFace/LLM/internlm)   |
| InternVL2   |   | [link](python/llm/example/GPU/HuggingFace/Multimodal/internvl2)   |
| Qwen       | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen)      | [link](python/llm/example/GPU/HuggingFace/LLM/qwen)       |
| Qwen1.5 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen1.5) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen1.5) |
| Qwen2 | [link](python/llm/example/CPU/HF-Transformers-AutoModels/Model/qwen2) | [link](python/llm/example/GPU/HuggingFace/LLM/qwen2) |
| Qwen2.5 |  | [link](python/llm/example/GPU/HuggingFace/LLM/qwen2.5) |
| Qwen-VL    | [link](python/llm/example
