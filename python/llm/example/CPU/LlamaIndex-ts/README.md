# LlamaIndex-ts

This folder contains examples showcasing how to use [**LlamaIndex-ts**](https://ts.llamaindex.ai/guides/agents/setup) with `ipex-llm`.
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
* Origin Ollama: refer to [this page](https://github.com/ollama/ollama) to install Ollama on CPU. 
* Ollama enabled by IPEX-LLM: you can use commands below to install and run ollama. 
    ```
    pip install --pre --upgrade ipex-llm[cpp]
    init-ollama
    ```

## Examples

### Agent

Run the `agent.ts` by `npx tsx agent.ts`. Agent calls the tool and summarizes the result. You will get similar output like below. 
```
{
  toolCall: {
    id: 'c5301d53-ede6-4492-b883-7b4165f07e1f',
    input: { a: 100, b: 201 },
    name: 'sumNumbers'
  }
}
{
  toolCall: {
    id: 'c5301d53-ede6-4492-b883-7b4165f07e1f',
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
      content: '"""" \nThought: The result of the sum is 301.\nAnswer: 301\n""""'
    },
    raw: {
      model: 'llama3:latest',
      created_at: '2024-06-12T05:16:13.927097556Z',
      message: [Object],
      done: true,
      total_duration: 3808496008,
      load_duration: 132690,
      prompt_eval_count: 53,
      prompt_eval_duration: 2017259000,
      eval_count: 22,
      eval_duration: 1789000000
    }
  },
  sources: [Getter]
}
```