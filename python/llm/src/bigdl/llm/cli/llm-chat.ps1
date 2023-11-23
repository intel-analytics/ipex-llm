$llm_dir = (Split-Path -Parent (python -c "import bigdl.llm;print(bigdl.llm.__file__)"))
$lib_dir = Join-Path $llm_dir "libs"
$prompt_dir = Join-Path $llm_dir "cli/prompts"

$vnni_enable = ((python -c "from bigdl.llm.utils.isa_checker import check_avx_vnni;print(check_avx_vnni())").ToLower() -eq "true")
$model_family = ""
$threads = 8
# Number of tokens to predict (made it larger than default because we want a long interaction)
$n_predict = 512

# Function to display help message
function Display-Help
{
    Write-Host "usage: ./llm-cli.ps1 -x MODEL_FAMILY [-h] [args]"
    Write-Host ""
    Write-Host "options:"
    Write-Host "  -h, --help           show this help message"
    Write-Host "  -x, --model_family {llama,bloom,gptneox}"
    Write-Host "                       family name of model"
    Write-Host "  -t N, --threads N    number of threads to use during computation (default: 8)"
    Write-Host "  -n N, --n_predict N  number of tokens to predict (default: 128, -1 = infinity)"
    Write-Host "  args                 parameters passed to the specified model function"
}

function llama
{
    $exec_file = "main-llama.exe"
    $prompt_file = Join-Path $prompt_dir "chat-with-llm.txt"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict -f $prompt_file -i --color --reverse-prompt 'USER:' --in-prefix ' ' $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

function gptneox
{
    $exec_file = "main-gptneox.exe"
    $prompt = "A chat between a curious human and an artificial intelligence assistant.`
            The assistant gives helpful, detailed, and polite answers."
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict --color --instruct -p '$prompt' $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

# Remove model_family/x parameter
$filteredArguments = @()
for ($i = 0; $i -lt $args.Length; $i++) {
    if ($args[$i] -eq '--model_family' -or $args[$i] -eq '--model-family' -or $args[$i] -eq '-x')
    {
        if ($i + 1 -lt $args.Length -and $args[$i + 1] -notlike '-*')
        {
            $i++
            $model_family = $args[$i]
        }
    }
    elseif ($args[$i] -eq '--threads' -or $args[$i] -eq '-t')
    {
        $i++
        $threads = $args[$i]
    }
    elseif ($args[$i] -eq '--n_predict' -or $args[$i] -eq '--n-predict' -or $args[$i] -eq '-n')
    {
        $i++
        $n_predict = $args[$i]
    }
    else
    {
        $filteredArguments += "`'" + $args[$i] + "`'"
    }
}

# Perform actions based on the model_family
switch ($model_family)
{
    "llama" {
        llama
    }
    "gptneox" {
        gptneox
    }
    default {
        Write-Host "llm-chat does not support model_family $model_family for now."
        Display-Help
    }
}
