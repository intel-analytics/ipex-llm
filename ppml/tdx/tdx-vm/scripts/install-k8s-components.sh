#!/bin/bash

#swapoff before installing kubernetes
#Install docker
#Configuration of kubeadm kubectl and kubelet 


sudo apt-get update

echo "....................swapoff"

swapoff -a

echo "....................Installing sshpass "

sudo apt-get install -y sshpass   

#echo "....................openssh server installation"
#sudo apt-get install -y openssh-server

sudo apt-get update
echo "....................Docker installtion"

sudo apt-get install -y docker.io
systemctl daemon-reload
systemctl restart docker
docker version
sudo docker run hello-world

echo ".....................Installing kubeadm kubelet kubectl"
curl https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg |sudo apt-key add -
sleep 30s
echo "deb http://mirrors.ustc.edu.cn/kubernetes/apt kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update

sudo apt-get install -y kubelet=1.23.4-00 kubeadm=1.23.4-00 kubectl=1.23.4-00

apt-mark hold kubelet kubeadm kubectl


echo ".......................Restart kubelet"

systemctl daemon-reload
systemctl restart kubelet

sudo apt-get update

echo ".......................Execute sucessfully"
