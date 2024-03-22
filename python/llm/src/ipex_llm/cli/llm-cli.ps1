$llm_dir = (Split-Path -Parent (python -c "import ipex_llm;print(ipex_llm.__file__)"))
$lib_dir = Join-Path $llm_dir "libs"


$vnni_enable = ((python -c "from ipex_llm.utils.isa_checker import check_avx_vnni;print(check_avx_vnni())").ToLower() -eq "true")
$model_family = ""
$threads = 8
$n_predict = 128

# Function to display help message
function Display-Help
{
    Write-Host "usage: ./llm-cli.ps1 -x MODEL_FAMILY [-h] [args]"
    Write-Host ""
    Write-Host "options:"
    Write-Host "  -h, --help           show this help message"
    Write-Host "  -x, --model_family {llama,bloom,gptneox,starcoder,chatglm}"
    Write-Host "                       family name of model"
    Write-Host "  -t N, --threads N    number of threads to use during computation (default: 8)"
    Write-Host "  -n N, --n_predict N  number of tokens to predict (default: 128, -1 = infinity)"
    Write-Host "  args                 parameters passed to the specified model function"
}

function llama
{
    $exec_file = "main-llama.exe"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

function bloom
{
    $exec_file = "main-bloom.exe"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

function gptneox
{
    $exec_file = "main-gptneox.exe"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

function starcoder
{
    $exec_file = "main-starcoder.exe"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict $filteredArguments"
    Write-Host "$command"
    Invoke-Expression $command
}

function chatglm
{
    $exec_file = "main-chatglm_vnni.exe"
    $command = "$lib_dir/$exec_file -t $threads -n $n_predict $filteredArguments"
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
    "bloom" {
        bloom
    }
    "gptneox" {
        gptneox
    }
    "starcoder" {
        starcoder
    }
    "chatglm" {
        chatglm
    }
    default {
        Write-Host "Invalid model_family: $model_family"
        Display-Help
    }
}
