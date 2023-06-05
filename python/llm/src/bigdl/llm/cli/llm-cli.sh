#!/bin/bash

# Default values
#interactive=false
#interactive_first=false
#instruct=false
#color=false
#random_prompt=false
#ignore_eos=false
#no_penalize_nl=false
#perplexity=false
#memory_f32=false
#mlock=false
#no_mmap=false
#mtest=false
#verbose_prompt=false

threads=4
n_predict=128
top_k=40
top_p=0.9
seed=-1
tfs=1.0
typical=1.0
repeat_last_n=64
repeat_penalty=1.1
presence_penalty=0.0
frequency_penalty=0.0
mirostat=0
mirostat_lr=0.1
mirostat_ent=5.0
ctx_size=512
temp=0.8
n_parts=-1
batch_size=512
keep=0
model="./model.bin"
model_family=""

llm_dir="$(dirname "$(python -c "import bigdl.llm;print(bigdl.llm.__file__)")")"
lib_dir="$llm_dir/libs"

# Function to display help message
function display_help {
  echo "usage: ./llm-cli.sh [options]"
  echo ""
  echo "options:"
  echo "  -h, --help                        show this help message and exit"
  echo "  -i, --interactive                 run in interactive mode"
  echo "  --interactive-first               run in interactive mode and wait for input right away"
  echo "  -ins, --instruct                  run in instruction mode"
  echo "  -r, --reverse-prompt PROMPT       run in interactive mode and poll user input upon seeing PROMPT"
  echo "                                    (can be specified more than once for multiple prompts)."
  echo "  --color                           colorise output to distinguish prompt and user input from generations"
  echo "  -s, --seed SEED                   RNG seed (default: -1, use random seed for <= 0)"
  echo "  -t, --threads N                   number of threads to use during computation (default: 4)"
  echo "  -p, --prompt PROMPT               prompt to start generation with (default: empty)"
  echo "  --session FNAME                   file to cache model state in (may be large!) (default: none)"
  echo "  --random-prompt                   start with a randomized prompt."
  echo "  --in-prefix STRING                string to prefix user inputs with (default: empty)"
  echo "  -f, --file FNAME                  prompt file to start generation."
  echo "  -n, --n_predict N                 number of tokens to predict (default: 128, -1 = infinity)"
  echo "  --top_k N                         top-k sampling (default: 40, 0 = disabled)"
  echo "  --top_p N                         top-p sampling (default: 0.9, 1.0 = disabled)"
  echo "  --tfs N                           tail free sampling, parameter z (default: 1.0, 1.0 = disabled)"
  echo "  --typical N                       locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)"
  echo "  --repeat_last_n N                 last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)"
  echo "  --repeat_penalty N                penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)"
  echo "  --presence_penalty N              repeat alpha presence penalty (default: 0.0, 0.0 = disabled)"
  echo "  --frequency_penalty N             repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)"
  echo "  --mirostat N                      use Mirostat sampling."
  echo "                                    Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used."
  echo "                                    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"
  echo "  --mirostat_lr N                   Mirostat learning rate, parameter eta (default: 0.1)"
  echo "  --mirostat_ent N                  Mirostat target entropy, parameter tau (default: 5.0)"
  echo "  -l, --logit-bias TOKEN_ID(+/-)BIAS modifies the likelihood of token appearing in the completion,"
  echo "                                    i.e. '--logit-bias 15043+1' to increase likelihood of token ' Hello',"
  echo "                                    or '--logit-bias 15043-1' to decrease likelihood of token ' Hello'"
  echo "  -c, --ctx_size N                  size of the prompt context (default: 512)"
  echo "  --ignore-eos                      ignore end of stream token and continue generating (implies --logit-bias 2-inf)"
  echo "  --no-penalize-nl                  do not penalize newline token"
  echo "  --memory_f32                      use f32 instead of f16 for memory key+value"
  echo "  --temp N                          temperature (default: 0.8)"
  echo "  --n_parts N                       number of model parts (default: -1 = determine from dimensions)"
  echo "  -b, --batch_size N                batch size for prompt processing (default: 512)"
  echo "  --perplexity                      compute perplexity over the prompt"
  echo "  --keep N                          number of tokens to keep from the initial prompt (default: 0, -1 = all)"
  echo "  --mlock                           force system to keep model in RAM rather than swapping or compressing"
  echo "  --no-mmap                         do not memory-map model (slower load but may reduce pageouts if not using mlock)"
  echo "  --mtest                           compute maximum memory usage"
  echo "  --verbose-prompt                  print prompt before generation"
  echo "  --lora FNAME                      apply LoRA adapter (implies --no-mmap)"
  echo "  --lora-base FNAME                 optional model to use as a base for the layers modified by the LoRA adapter"
  echo "  -m, --model FNAME                 model path (default: ./model.bin)"
  echo "  -x, --model_family {llama,bloom,gptneox} family name of model"
}

function llama {
  echo "Unsupported params:"
  [ -n "$session" ] && echo " session: $session"
  [ -n "$tfs" ] && echo " tfs: $tfs"
  [ -n "$typical" ] && echo " typical: $typical"
  [ -n "$presence_penalty" ] && echo " presence_penalty: $presence_penalty"
  [ -n "$frequency_penalty" ] && echo " frequency_penalty: $frequency_penalty"
  [ -n "$mirostat" ] && echo " mirostat: $mirostat"
  [ -n "$mirostat_lr" ] && echo " mirostat_lr: $mirostat_lr"
  [ -n "$mirostat_ent" ] && echo " mirostat_ent: $mirostat_ent"
  [ -n "$logit_bias" ] && echo " logit_bias: $logit_bias"
  [ -n "$no_penalize_nl" ] && echo " no_penalize_nl: $no_penalize_nl"
  command="$lib_dir/main-llama \
    -s $seed \
    -t $threads \
    -n $n_predict \
    --top_k $top_k \
    --top_p $top_p \
    --repeat_last_n $repeat_last_n \
    --repeat_penalty $repeat_penalty \
    -c $ctx_size \
    --temp $temp \
    --n_parts $n_parts \
    -b $batch_size \
    --keep $keep \
    -m $model \
    ${interactive:+-i} \
    ${interactive_first:+--interactive-first} \
    ${instruct:+--instruct} \
    ${reverse_prompt:+-r "$reverse_prompt"} \
    ${color:+--color} \
    ${prompt:+-p "$prompt"} \
    ${random_prompt:+--random-prompt} \
    ${in_prefix:+--in-prefix "$in_prefix"} \
    ${file:+-f "$file"} \
    ${ignore_eos:+--ignore-eos} \
    ${memory_f32:+--memory_f32} \
    ${perplexity:+--perplexity} \
    ${mlock:+--mlock} \
    ${no_mmap:+--no-mmap} \
    ${mtest:+--mtest} \
    ${verbose_prompt:+--verbose-prompt} \
    ${lora:+--lora "$lora"} \
    ${lora_base:+--lora-base "$lora_base"}"
  echo "$command"
  $command
}

function bloom {
  echo "Unsupported params:"
  [ -n "$interactive" ] && echo " interactive: $interactive"
  [ -n "$interactive_first" ] && echo " interactive_first: $interactive_first"
  [ -n "$instruct" ] && echo " instruct: $instruct"
  [ -n "$color" ] && echo " color: $color"
  [ -n "$session" ] && echo " session: $session"
  [ -n "$random_prompt" ] && echo " random_prompt: $random_prompt"
  [ -n "$in_prefix" ] && echo " in_prefix: $in_prefix"
  [ -n "$file" ] && echo " file: $file"
  [ -n "$tfs" ] && echo " tfs: $tfs"
  [ -n "$typical" ] && echo " typical: $typical"
  [ -n "$presence_penalty" ] && echo " presence_penalty: $presence_penalty"
  [ -n "$frequency_penalty" ] && echo " frequency_penalty: $frequency_penalty"
  [ -n "$mirostat" ] && echo " mirostat: $mirostat"
  [ -n "$mirostat_lr" ] && echo " mirostat_lr: $mirostat_lr"
  [ -n "$mirostat_ent" ] && echo " mirostat_ent: $mirostat_ent"
  [ -n "$logit_bias" ] && echo " logit_bias: $logit_bias"
  [ -n "$ctx_size" ] && echo " ctx_size: $ctx_size"
  [ -n "$ignore_eos" ] && echo " ignore_eos: $ignore_eos"
  [ -n "$no_penalize_nl" ] && echo " no_penalize_nl: $no_penalize_nl"
  [ -n "$memory_f32" ] && echo " memory_f32: $memory_f32"
  [ -n "$n_parts" ] && echo " n_parts: $n_parts"
  [ -n "$perplexity" ] && echo " perplexity: $perplexity"
  [ -n "$keep" ] && echo " keep: $keep"
  [ -n "$mlock" ] && echo " mlock: $mlock"
  [ -n "$no_mmap" ] && echo " no_mmap: $no_mmap"
  [ -n "$mtest" ] && echo " mtest: $mtest"
  [ -n "$verbose_prompt" ] && echo " verbose_prompt: $verbose_prompt"
  [ -n "$lora" ] && echo " lora: $lora"
  [ -n "$memory_f32" ] && echo " memory_f32: $memory_f32"
  [ -n "$lora_base" ] && echo " lora_base: $lora_base"
  command="$lib_dir/main-bloom \
    -s $seed \
    -t $threads \
    -n $n_predict \
    --top_k $top_k \
    --top_p $top_p \
    --repeat_last_n $repeat_last_n \
    --repeat_penalty $repeat_penalty \
    --temp $temp \
    -b $batch_size \
    -m $model \
    ${prompt:+-p "$prompt"}"
  echo "$command"
  $command
}

function gptneox {
  command="$lib_dir/main-gptneox \
    -s $seed \
    -t $threads \
    -n $n_predict \
    --top_k $top_k \
    --top_p $top_p \
    --tfs $tfs \
    --typical $typical \
    --repeat_last_n $repeat_last_n \
    --repeat_penalty $repeat_penalty \
    --presence_penalty $presence_penalty \
    --frequency_penalty $frequency_penalty \
    --mirostat $mirostat \
    --mirostat_lr $mirostat_lr \
    --mirostat_ent $mirostat_ent \
    -c $ctx_size \
    --temp $temp \
    --n_parts $n_parts \
    -b $batch_size \
    --keep $keep \
    -m $model \
    ${interactive:+-i} \
    ${interactive_first:+--interactive-first} \
    ${instruct:+--instruct} \
    ${reverse_prompt:+-r "$reverse_prompt"} \
    ${color:+--color} \
    ${prompt:+-p "$prompt"} \
    ${session:+--session "$session"} \
    ${random_prompt:+--random-prompt} \
    ${in_prefix:+--in-prefix "$in_prefix"} \
    ${file:+-f "$file"} \
    ${logit_bias:+-l "$logit_bias"} \
    ${ignore_eos:+--ignore-eos} \
    ${no_penalize_nl:+--no-penalize-nl} \
    ${memory_f32:+--memory_f32} \
    ${perplexity:+--perplexity} \
    ${mlock:+--mlock} \
    ${no_mmap:+--no-mmap} \
    ${mtest:+--mtest} \
    ${verbose_prompt:+--verbose-prompt} \
    ${lora:+--lora "$lora"} \
    ${lora_base:+--lora-base "$lora_base"}"
  echo "$command"
  $command
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
  -h | --help)
    display_help
    exit 0
    ;;
  -i | --interactive)
    interactive=true
    shift
    ;;
  --interactive-first)
    interactive_first=true
    shift
    ;;
  -ins | --instruct)
    instruct=true
    shift
    ;;
  -r | --reverse-prompt)
    reverse_prompt="$2"
    shift 2
    ;;
  --color)
    color=true
    shift
    ;;
  -s | --seed)
    seed="$2"
    shift 2
    ;;
  -t | --threads)
    threads="$2"
    shift 2
    ;;
  -p | --prompt)
    prompt="$2"
    shift 2
    ;;
  --session)
    session="$2"
    shift 2
    ;;
  --random-prompt)
    random_prompt=true
    shift
    ;;
  --in-prefix)
    in_prefix="$2"
    shift 2
    ;;
  -f | --file)
    file="$2"
    shift 2
    ;;
  -n | --n_predict)
    n_predict="$2"
    shift 2
    ;;
  --top_k)
    top_k="$2"
    shift 2
    ;;
  --top_p)
    top_p="$2"
    shift 2
    ;;
  --tfs)
    tfs="$2"
    shift 2
    ;;
  --typical)
    typical="$2"
    shift 2
    ;;
  --repeat_last_n)
    repeat_last_n="$2"
    shift 2
    ;;
  --repeat_penalty)
    repeat_penalty="$2"
    shift 2
    ;;
  --presence_penalty)
    presence_penalty="$2"
    shift 2
    ;;
  --frequency_penalty)
    frequency_penalty="$2"
    shift 2
    ;;
  --mirostat)
    mirostat="$2"
    shift 2
    ;;
  --mirostat_lr)
    mirostat_lr="$2"
    shift 2
    ;;
  --mirostat_ent)
    mirostat_ent="$2"
    shift 2
    ;;
  -l | --logit-bias)
    logit_bias="$2"
    shift 2
    ;;
  -c | --ctx_size)
    ctx_size="$2"
    shift 2
    ;;
  --ignore-eos)
    ignore_eos=true
    shift
    ;;
  --no-penalize-nl)
    no_penalize_nl=true
    shift
    ;;
  --memory_f32)
    memory_f32=true
    shift
    ;;
  --temp)
    temp="$2"
    shift 2
    ;;
  --n_parts)
    n_parts="$2"
    shift 2
    ;;
  -b | --batch_size)
    batch_size="$2"
    shift 2
    ;;
  --perplexity)
    perplexity=true
    shift
    ;;
  --keep)
    keep="$2"
    shift 2
    ;;
  --mlock)
    mlock=true
    shift
    ;;
  --no-mmap)
    no_mmap=true
    shift
    ;;
  --mtest)
    mtest=true
    shift
    ;;
  --verbose-prompt)
    verbose_prompt=true
    shift
    ;;
  --lora)
    lora="$2"
    shift 2
    ;;
  --lora-base)
    lora_base="$2"
    shift 2
    ;;
  -m | --model)
    model="$2"
    shift 2
    ;;
  -x | --model_family)
    model_family="$2"
    shift 2
    ;;
  *)
    echo "Invalid option: $1"
    display_help
    exit 1
    ;;
  esac
done

# Print the values of the parsed arguments
#echo "model_family: $model_family"
#echo "model: $model"
#echo "threads: $threads"
#echo "prompt: $prompt"
#echo "n_predict: $n_predict"
#echo "interactive: $interactive"
#echo "reverse_prompt: $reverse_prompt"
#echo "interactive_first: $interactive_first"
#echo "instruct: $instruct"
#echo "color: $color"
#echo "seed: $seed"
#echo "session: $session"
#echo "random_prompt: $random_prompt"
#echo "in_prefix: $in_prefix"
#echo "file: $file"
#echo "top_k: $top_k"
#echo "top_p: $top_p"
#echo "tfs: $tfs"
#echo "typical: $typical"
#echo "repeat_last_n: $repeat_last_n"
#echo "repeat_penalty: $repeat_penalty"
#echo "presence_penalty: $presence_penalty"
#echo "frequency_penalty: $frequency_penalty"
#echo "mirostat: $mirostat"
#echo "mirostat_lr: $mirostat_lr"
#echo "mirostat_ent: $mirostat_ent"
#echo "logit_bias: $logit_bias"
#echo "ctx_size: $ctx_size"
#echo "ignore_eos: $ignore_eos"
#echo "no_penalize_nl: $no_penalize_nl"
#echo "memory_f32: $memory_f32"
#echo "temp: $temp"
#echo "n_parts: $n_parts"
#echo "batch_size: $batch_size"
#echo "perplexity: $perplexity"
#echo "keep: $keep"
#echo "mlock: $mlock"
#echo "no_mmap: $no_mmap"

# Perform actions based on the model_family
if [[ "$model_family" == "llama" ]]; then
  llama
elif [[ "$model_family" == "bloom" ]]; then
  bloom
elif [[ "$model_family" == "gptneox" ]]; then
  gptneox
else
  echo "Invalid model_family: $model_family"
  display_help
fi
