#!/bin/bash
set -e

master_server='172.168.3.116'
master_port='6379'
slave_server_list=('172.168.3.116' '172.168.3.118')
slave_port_list=('6380' '6381' '6382' '6383' '6384' '6385' '6386' '6387')
log_path="~/redis"
cmd_list=""

echo "Starting master...."
ssh $master_server "redis-server --port $master_port --protected-mode no --dir $log_path --logfile $log_path/$master_port.log --daemonize yes"

for p in "${slave_port_list[@]}"; do
    cmdlist="${cmdlist}redis-server --port $p --protected-mode no --dir $log_path --logfile $log_path/$p.log --replicaof $master_server $master_port --daemonize yes;"
done

for s in "${slave_server_list[@]}"; do
    ssh $s "$cmdlist"
done
