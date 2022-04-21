[array]$wmi_cpu = Get-WmiObject -Class Win32_Processor
[int]$number_of_socket = $wmi_cpu.length
[int]$cores_per_socket = $wmi_cpu.NumberOfCores / $number_of_socket
[int]$numberLogicalProcessors = $wmi_cpu.NumberOfLogicalProcessors
[int]$threads_per_core = $numberLogicalProcessors / $wmi_cpu.NumberOfCores

$env:OMP_NUM_THREADS=$cores_per_socket*$number_of_socket
if ($threads_per_core -gt 1) {
    $env:KMP_AFFINITY="granularity=fine,compact,1,0"
}
else {
    $env:KMP_AFFINITY="granularity=fine,compact"
}
$env:KMP_BLOCKTIME=1
$env:TF_ENABLE_ONEDNN_OPTS=1
$env:NANO_TF_INTER_OP=1
Write-Host "==================Environment Variables================="
Write-Host "OMP_NUM_THREADS=${env:OMP_NUM_THREADS}"
Write-Host "KMP_AFFINITY=${env:KMP_AFFINITY}"
Write-Host "KMP_BLOCKTIME=${env:KMP_BLOCKTIME}"
Write-Host "TF_ENABLE_ONEDNN_OPTS=${env:TF_ENABLE_ONEDNN_OPTS}"
Write-Host "NANO_TF_INTER_OP=${env:NANO_TF_INTER_OP}"
Write-Host "========================================================="
