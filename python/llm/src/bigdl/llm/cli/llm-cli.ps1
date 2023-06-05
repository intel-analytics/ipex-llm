$script_dir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$lib_dir = Join-Path -Path $script_dir -ChildPath "libs"

# Function to display help message
function Display-Help {
    Write-Host "usage: ./llm-cli.ps1 [options]"
    Write-Host ""
    Write-Host "options:"
    Write-Host "  -h, --help            show this help message and exit"
    Write-Host "  -i, --interactive     run in interactive mode"
    Write-Host "  --interactive-first   run in interactive mode and wait for input right away"
    Write-Host "  -ins, --instruct      run in instruction mode"
    Write-Host "  -r PROMPT, --reverse-prompt PROMPT"
    Write-Host "                        run in interactive mode and poll user input upon seeing PROMPT (can be"
    Write-Host "                        specified more than once for multiple prompts)."
    Write-Host "  --color               colorise output to distinguish prompt and user input from generations"
    Write-Host "  -s SEED, --seed SEED  RNG seed (default: -1, use random seed for <= 0)"
    Write-Host "  -t N, --threads N     number of threads to use during computation (default: 4)"
    Write-Host "  -p PROMPT, --prompt PROMPT"
    Write-Host "                        prompt to start generation with (default: empty)"
    Write-Host "  --session FNAME       file to cache model state in (may be large!) (default: none)"
    Write-Host "  --random-prompt       start with a randomized prompt."
    Write-Host "  --in-prefix STRING    string to prefix user inputs with (default: empty)"
    Write-Host "  -f FNAME, --file FNAME"
    Write-Host "                        prompt file to start generation."
    Write-Host "  -n N, --n_predict N   number of tokens to predict (default: 128, -1 = infinity)"
    Write-Host "  --top_k N             top-k sampling (default: 40, 0 = disabled)"
    Write-Host "  --top_p N             top-p sampling (default: 0.9, 1.0 = disabled)"
    Write-Host "  --tfs N               tail free sampling, parameter z (default: 1.0, 1.0 = disabled)"
    Write-Host "  --typical N           locally typical sampling, parameter p (default: 1.0, 1.0 = disabled)"
    Write-Host "  --repeat_last_n N     last n tokens to consider for penalize (default: 64, 0 = disabled, -1 = ctx_size)"
    Write-Host "  --repeat_penalty N    penalize repeat sequence of tokens (default: 1.1, 1.0 = disabled)"
    Write-Host "  --presence_penalty N  repeat alpha presence penalty (default: 0.0, 0.0 = disabled)"
    Write-Host "  --frequency_penalty N repeat alpha frequency penalty (default: 0.0, 0.0 = disabled)"
    Write-Host "  --mirostat N          use Mirostat sampling."
    Write-Host "                        Top K, Nucleus, Tail Free and Locally Typical samplers are ignored if used."
    Write-Host "                        (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"
    Write-Host "  --mirostat_lr N       Mirostat learning rate, parameter eta (default: 0.1)"
    Write-Host "  --mirostat_ent N      Mirostat target entropy, parameter tau (default: 5.0)"
    Write-Host "  -l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS"
    Write-Host "                        modifies the likelihood of token appearing in the completion,"
    Write-Host "                        i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',"
    Write-Host "                        or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'"
    Write-Host "  -c N, --ctx_size N    size of the prompt context (default: 512)"
    Write-Host "  --ignore-eos          ignore end of stream token and continue generating (implies --logit-bias 2-inf)"
    Write-Host "  --no-penalize-nl      do not penalize newline token"
    Write-Host "  --memory_f32          use f32 instead of f16 for memory key+value"
    Write-Host "  --temp N              temperature (default: 0.8)"
    Write-Host "  --n_parts N           number of model parts (default: -1 = determine from dimensions)"
    Write-Host "  -b N, --batch_size N  batch size for prompt processing (default: 512)"
    Write-Host "  --perplexity          compute perplexity over the prompt"
    Write-Host "  --keep                number of tokens to keep from the initial prompt (default: 0, -1 = all)"
    Write-Host "  --mlock               force system to keep model in RAM rather than swapping or compressing"
    Write-Host "  --no-mmap             do not memory-map model (slower load but may reduce pageouts if not using mlock)"
    Write-Host "  --mtest               compute maximum memory usage"
    Write-Host "  --verbose-prompt      print prompt before generation"
    Write-Host "  --lora FNAME          apply LoRA adapter (implies --no-mmap)"
    Write-Host "  --lora-base FNAME     optional model to use as a base for the layers modified by the LoRA adapter"
    Write-Host "  -m FNAME, --model FNAME"
    Write-Host "                        model path"
    Write-Host "  -x, --model_family {llama,bloom,gptneox}"
    Write-Host "                        family name of model"
}

function llama {
    $command = "$lib_dir/main-llama.exe $filteredArguments"
    Invoke-Expression $command
}

function bloom {
    $command = "$lib_dir/main-bloom.exe $filteredArguments"
    Invoke-Expression $command
}

function gptneox {
    $command = "$lib_dir/main-gptneox.exe $filteredArguments"
    Invoke-Expression $command
}

# Remove model_family/x parameter
$filteredArguments = @()
for ($i = 0; $i -lt $args.Length; $i++) {
    if ($args[$i] -eq '--model_family' -or $args[$i] -eq '-x') {
        if ($i + 1 -lt $args.Length -and $args[$i + 1] -notlike '-*') {
            $i++
            $model_family = $args[$i]
        }
    }
    else {
        $filteredArguments += $args[$i]
    }
}

# Perform actions based on the model_family
switch ($model_family) {
    "llama" { llama }
    "bloom" { bloom }
    "gptneox" { gptneox }
    default {
        Write-Host "Invalid model_family: $model_family"
        Display-Help
    }
}
