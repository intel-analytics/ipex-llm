#!/bin/bash
num=4096
dtype='int'

while getopts "n:t:" opt
do
    case $opt in
        n)
            num=$OPTARG
        ;;
        t)
            dtype=$OPTARG
        ;;
    esac
done

cd /ppml
./init.sh
bash examples/numpy/CLI.sh -n $num -p native -t $dtype
export sgx_command="bash examples/numpy/CLI.sh -n $num -p sgx -t $dtype"
gramine-sgx bash 2>&1

