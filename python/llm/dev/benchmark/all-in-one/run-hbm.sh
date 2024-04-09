#!/bin/bash
source ipex-llm-init -t

sockets_num=$(lscpu | grep "Socket(s)" | awk -F ':' '{print $2}')
cores_per_socket=$(lscpu | grep "Core(s) per socket" | awk -F ':' '{print $2}')
numa_nodes=$(lscpu | grep "NUMA node(s)" | awk -F ':' '{print $2}')
# Multiply by 2 to avoid an float result in HBM flat mode that the NUMA count twice and it will be divided later.
cores_per_numa=$(($sockets_num * $cores_per_socket * 2 / $numa_nodes))

# Only support Quad-mode now
if [ "${numa_nodes}" -eq 4 ]; then
    #HBM flat Quad-mode, Confirm that there are 2 HBM memory nodes and 2 DRAM memory nodes through "nuamctl -H"
    echo "HBM Quad mode"
    export OMP_NUM_THREADS=${cores_per_numa}
    echo "OMP_NUM_THREADS: ${cores_per_numa}"
    last_cpu_index=$(($OMP_NUM_THREADS - 1))
    numactl -C 0-$last_cpu_index -p 2 python $(dirname "$0")/run.py
elif [ "${numa_nodes}" -eq 2 ]; then
    #SPR or hbm only or hbm cache Quad-mode, Confirm that there are 2 DRAM memory nodes through "nuamctl -H"
    echo "Warning: SPR Quad mode, hbm usage is default off, please check if HBM can be on."
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    echo "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    last_cpu_index=$(($OMP_NUM_THREADS - 1))
    numactl -C 0-$last_cpu_index -p 0 python $(dirname "$0")/run.py
elif [ "${numa_nodes}" -eq 1 ]; then
    # General Test mode
    echo "General Test mode"
    export OMP_NUM_THREADS=$((${cores_per_numa} / 2))
    echo "OMP_NUM_THREADS: $((${cores_per_numa} / 2))"
    last_cpu_index=$(($OMP_NUM_THREADS - 1))
    numactl -C 0-$last_cpu_index -p 0 python $(dirname "$0")/run.py
else
    echo "Warning: The number of nodes in this machine is ${numa_nodes}. Node 0 will be used for run. "
    export OMP_NUM_THREADS=${cores_per_numa}
    echo "OMP_NUM_THREADS: ${cores_per_numa}"
    last_cpu_index=$(($OMP_NUM_THREADS - 1))
    numactl -C 0-$last_cpu_index -p 0 python $(dirname "$0")/run.py
fi
