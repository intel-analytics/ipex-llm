#!/bin/bash

dataset="/ppml/work/data/dataset.csv"

while getopts "d:" opt
do
    case $opt in
        d)
            dataset=$OPTARG
        ;;
    esac
done

cd /ppml
./init.sh
bash /ppml/examples/pandas/CLI.sh -d $dataset -p native
export sgx_command="bash /ppml/examples/pandas/CLI.sh -d $dataset -p sgx"
gramine-sgx bash 2>&1

