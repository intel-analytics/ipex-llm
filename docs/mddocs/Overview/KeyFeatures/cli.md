# CLI (Command Line Interface) Tool

> [!NOTE] 
> Currently `ipex-llm` CLI supports *LLaMA* (e.g., vicuna), *GPT-NeoX* (e.g., redpajama), *BLOOM* (e.g., pheonix) and *GPT2* (e.g., starcoder) model architecture; for other models, you may use the `transformers`-style or LangChain APIs.

## Convert Model

You may convert the downloaded model into native INT4 format using `llm-convert`.

```bash
# convert PyTorch (fp16 or fp32) model; 
# llama/bloom/gptneox/starcoder model family is currently supported
llm-convert "/path/to/model/" --model-format pth --model-family "bloom" --outfile "/path/to/output/"

# convert GPTQ-4bit model
# only llama model family is currently supported
llm-convert "/path/to/model/" --model-format gptq --model-family "llama" --outfile "/path/to/output/"
```

## Run Model

You may run the converted model using `llm-cli` or `llm-chat` (built on top of `main.cpp` in [`llama.cpp`](https://github.com/ggerganov/llama.cpp))

```bash
# help
# llama/bloom/gptneox/starcoder model family is currently supported
llm-cli -x gptneox -h

# text completion
# llama/bloom/gptneox/starcoder model family is currently supported
llm-cli -t 16 -x gptneox -m "/path/to/output/model.bin" -p 'Once upon a time,'

# chat mode
# llama/gptneox model family is currently supported
llm-chat -m "/path/to/output/model.bin" -x llama
```