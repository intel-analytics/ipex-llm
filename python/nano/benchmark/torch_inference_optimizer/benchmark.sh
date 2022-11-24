process_number=$1
cores_per_process=$2
option=$3

unset KMP_AFFINITY

log_dir='benchmark_logs'
mkdir -p $log_dir
logs=""

for ((i=0; i<$process_number; i++)); do
    begin=$(($cores_per_process * i))
    end=$(($cores_per_process * (i+1) - 1))
    log_file=$log_dir/$i.log
    export OMP_NUM_THREADS=$cores_per_process
    echo "[$((i+1))/$process_number] cores: $begin-$end, OMP_NUM_THREADS: $OMP_NUM_THREADS, log_file: $log_file"
    taskset -c $begin-$end 1>$log_file 2>&1 \
        python inference.py --option $option &
    logs="$logs $log_file"
    # sleep 0.1
done

wait

throughput=$(grep 'Throughput:' $logs | sed -e 's/.*Throughput//;s/[^0-9.]//g' | awk -v i=$process_number '
    BEGIN {
        num = 0;
        sum = 0;
    }{
        num = num + 1;
        sum = sum + $1;
    }END {
        if(num > 0) {
            avg = sum / num;
            sum = avg * i;
        }
        printf("%.2f", sum);
    }
')

echo '>>>{"config": "'$option'", "throughput": '$throughput', "process": '$process_number'}<<<'
