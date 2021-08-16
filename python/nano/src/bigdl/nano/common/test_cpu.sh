total_cpu_cores=$(nproc)
number_sockets=$(($(grep "^physical id" /proc/cpuinfo | awk '{print $4}' | sort -un | tail -1)+1))
number_cpu_cores=$(( (total_cpu_cores/2) / number_sockets))

echo "number of CPU cores per socket: $number_cpu_cores"
echo "number of socket: $number_sockets"