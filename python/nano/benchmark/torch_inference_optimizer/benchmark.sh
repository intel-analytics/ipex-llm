#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


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
