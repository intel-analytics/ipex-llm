#!/bin/bash
cd /ppml
i=16384
n=32768
while [[ $i -le $n ]]
do
	bash examples/numpy/CLI.sh -n $i -p native -t float
	export sgx_command="bash examples/numpy/CLI.sh -n $i -p sgx -t float"
	gramine-sgx bash 2>&1
	i=$(( $i *2 ))
done

i=2048
n=4096
while [[ $i -le $n ]]
do
        bash examples/numpy/CLI.sh -n $i -p native -t int
        export sgx_command="bash examples/numpy/CLI.sh -n $i -p sgx -t int"
        gramine-sgx bash 2>&1
        i=$(( $i *2 ))
done
