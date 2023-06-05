param (
    [string[]]$args
)

    $ctx_size = 512
    $interactive = $false
    $n_predict = -1
    $model = ""
    $prompt = ""
    $threads = 28
    $model_family = ""

$script_dir = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$lib_dir = Join-Path -Path $script_dir -ChildPath "libs"

function DisplayHelp {
    Write-Host "Usage: ./llm-cli.ps1 [OPTIONS]"
    Write-Host "Options:"
    Write-Host "  -c, --ctx_size N      size of the prompt context (default: 512)"
    Write-Host "  -i, --interactive     run in interactive mode"
    Write-Host "  -n, --n_predict N     number of tokens to predict (default: -1, -1 = infinity)"
    Write-Host "  -m, --model FNAME     model path"
    Write-Host "  -p, --prompt PROMPT   prompt to start generation with (default: empty)"
    Write-Host "  -t, --threads N       number of threads to use during computation (default: 28)"
    Write-Host "  -x, --model_family {llama,bloom,gptneox}"
    Write-Host "                        family name of model"
    Write-Host "  -h, --help            display this help message"
}

function llama {
    $command = "$lib_dir/main-llama.exe -c $ctx_size -n $n_predict -p $prompt -t $threads"
    if ($interactive) {
        $command += " -i"
    }
    Invoke-Expression $command
}

function bloom {
    $command = "$lib_dir/main-bloom.exe -c $ctx_size -n $n_predict -p $prompt -t $threads"
    if ($interactive) {
        Write-Host "bloom model family does not support interactive mode"
        exit 1
    }
    Invoke-Expression $command
}

function gptneox {
    $command = "$lib_dir/main-gptneox.exe -c $ctx_size -n $n_predict -p $prompt -t $threads"
    if ($interactive) {
        $command += " -i"
    }
    Invoke-Expression $command
}



    $i = 0
    while ($i -lt $args.Length) {
        $arg = $args[$i]
        switch -wildcard ($arg) {
            "-c*", "--ctx_size" {
                $ctx_size = [int]$args[$i + 1]
                $i += 2
            }
            "-i*", "--interactive" {
                $interactive = $true
                $i++
            }
            "-n*", "--n_predict" {
                $n_predict = [int]$args[$i + 1]
                $i += 2
            }
            "-m*", "--model" {
                $model = $args[$i + 1]
                $i += 2
            }
            "-p*", "--prompt" {
                $prompt = $args[$i + 1]
                $i += 2
            }
            "-t*", "--threads" {
                $threads = [int]$args[$i + 1]
                $i += 2
            }
            "-x*", "--model_family" {
                $model_family = $args[$i + 1]
                $i += 2
            }
            "-h*", "--help" {
                DisplayHelp
                return
            }
            default {
                Write-Host "Invalid option: $($args[$i])"
                DisplayHelp
                return
            }
        }
    }

# Print the values of the parsed arguments
Write-Host "ctx_size: $ctx_size"
Write-Host "interactive: $interactive"
Write-Host "n_predict: $n_predict"
Write-Host "model: $model"
Write-Host "prompt: $prompt"
Write-Host "threads: $threads"
Write-Host "model_family: $model_family"

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
