# BigDL LLM

## llm-cli

llm-cli is a command-line interface tool that allows easy execution of llama/gptneox/bloom models
and generates results based on the provided prompt.

### Usage

```bash
llm-cli -x <llama/gptneox/bloom> [-h] [args]
```

`args` are the arguments provided to the specified model program. You can use `-x MODEL_FAMILY -h`
to retrieve the parameter list for a specific `MODEL_FAMILY`, for example:

```bash
llm-cli.sh -x llama -h

# Output:
# usage: main-llama [options]
#
# options:
#   -h, --help show this help message and exit
#   -i, --interactive run in interactive mode
#   --interactive-first run in interactive mode and wait for input right away
#   ...
```

### Examples

Here are some examples of how to use the llm-cli tool:

#### Completion:

```bash
llm-cli.sh -t 16 -x llama -m ./llm-llama-model.bin -p 'Once upon a time,'
```

#### Chatting:

```bash
llm-cli.sh -t 16 -x llama -m ./llm-llama-model.bin -i --color
```

Feel free to explore different options and experiment with the llama/gptneox/bloom models using
llm-cli!