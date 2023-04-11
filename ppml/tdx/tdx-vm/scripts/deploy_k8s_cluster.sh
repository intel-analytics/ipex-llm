#!/bin/bash

masterip=
slavepass=
slavepass2=
slaveip=
slaveip2=

echo "export KUBECONFIG=/etc/kubernetes/admin.conf" >> /etc/profile
source /etc/profile
echo '{"exec-opts": ["native.cgroupdriver=systemd"]}' | sudo tee /etc/docker/daemon.json
systemctl daemon-reload
systemctl restart docker
systemctl restart kubelet
kubeadm reset -f

sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --ignore-preflight-errors=NumCPU --image-repository=registry.aliyuncs.com/google_containers | tee -a output.txt

sudo export KUBECONFIG=/etc/kubernetes/admin.conf

echo "........................Network plugin "

kubectl apply -f /root/kube-flannel.yml


kubectl taint nodes --all node-role.kubernetes.io/master-

echo ".......................Joining node to cluster"

#output send to result file and that send to Knode folder

#Kubeadm init command output take into output.txt file and that convert to output.sh file and execute as shell

grep -A 1 "kubeadm join" ./output.txt | tee -a ./output.sh
chmod +x ./output.sh

echo "Running on slave machine"
sshpass -p $slavepass scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null output.sh  root@$slaveip:/root

sshpass -p $slavepass ssh -o StrictHostKeyChecking=no  root@$slaveip 'sh /root/output.sh'

sshpass -p $slavepass2 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null output.sh  root@$slaveip2:/root

sshpass -p $slavepass2 ssh -o StrictHostKeyChecking=no  root@$slaveip2 'sh /root/output.sh'


echo "......................._remove file content otherwise it create ambiguity-"

#>output.sh
#>output.txt

sleep 30s

kubectl get nodes
