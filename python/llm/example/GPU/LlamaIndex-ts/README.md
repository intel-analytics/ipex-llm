# LlamaIndex-ts

This folder contains examples showcasing how to use [**LlamaIndex-ts**](https://ts.llamaindex.ai/guides/agents/setup) with `ipex-llm` on Intel GPU.
> [**LlamaIndex-ts**](https://ts.llamaindex.ai/guides/agents/setup) helps you unlock domain-specific data and then build powerful applications with it.

## Setting up Dependencies 

### Install LlamaIndex-ts
Following [this page](https://github.com/intel-analytics/ipex-llm/blob/main/python/llm/example/CPU/LlamaIndex/README.md) to create an environment for llamaindex.
Then run the commands below
```
npm install llamaindex
npm install dotenv
```

### Install Ollama
* Ollama enabled by IPEX-LLM: you can use commands below to install and run ollama on Intel GPU locally. 
    ```
    pip install --pre --upgrade ipex-llm[cpp]
    init-ollama
    OLLAMA_HOST=0.0.0.0 ./ollama serve
    ```

## Examples

### Agent

Run the [agent.ts](./agent.ts) by `npx tsx agent.ts`. Agent calls the tool and summarizes the result. You will get similar output like below. 
```
{
  toolCall: {
    id: '82e6c885-8907-4650-a30b-c7ecd10f5c63',
    input: { a: 100, b: 201 },
    name: 'sumNumbers'
  }
}
{
  toolCall: {
    id: '82e6c885-8907-4650-a30b-c7ecd10f5c63',
    input: { a: 100, b: 201 },
    name: 'sumNumbers'
  },
  toolResult: {
    tool: FunctionTool { _fn: [Function: sumNumbers], _metadata: [Object] },
    input: { a: 100, b: 201 },
    output: '301',
    isError: false
  }
}
{
  response: {
    message: {
      role: 'assistant',
      content: "Thought: The result is correct, it's indeed 301. I can confirm that using the sumNumbers tool was accurate.\n" +
        '\n' +
        'Answer:\n' +
        '301'
    },
    raw: {
      model: 'llama3:latest',
      created_at: '2024-06-12T06:30:47.517156181Z',
      message: [Object],
      done_reason: 'stop',
      done: true,
      total_duration: 966566699,
      load_duration: 42646686,
      prompt_eval_count: 37,
      prompt_eval_duration: 321809000,
      eval_count: 29,
      eval_duration: 465253000
    }
  },
  sources: [Getter]
}
```