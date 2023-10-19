Write-Host "Please enter the model path: "
$modelpath = Read-Host
$use_new_transformers = 0

# execute chat script
$env:PYTHONUNBUFFERED=1

$json = Get-Content "$modelpath\config.json" | ConvertFrom-Json
if ($json.model_type -eq 'mistral') {
    $use_new_transformers = 1
}

if ($use_new_transformers -eq 1) {
    Write-Host "Mistral model detected, installing transformers==4.34.0"
    .\python-embed\python.exe -m pip install transformers==4.34.0 > $null
    if ($?) {
        Write-Host "Successfully installed transformers==4.34.0, now load model..."
    } else {
        Write-Host "Installed transformers==4.34.0 failed. exiting..."
        exit 1
    }
}

.\python-embed\python.exe .\chat.py --model-path=`"$modelpath`"

if($use_new_transformers -eq 1) {
    Write-Host "Rolling back to transformers==4.31.0"
    .\python-embed\python.exe -m pip install transformers==4.31.0 > $null
    if ($?) {
        Write-Host "Successfully installed transformers==4.31.0, now exit..."
    } else {
        Write-Host "Installed transformers==4.31.0 failed. exiting..."
        exit 1
    }
}

Read-Host -Prompt "Press Enter to exit"
