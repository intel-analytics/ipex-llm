#!/bin/bash

sudo apt install dkms
wget --no-check-certificate https://download.01.org/intel-sgx/sgx-linux/2.12/distro/ubuntu18.04-server/sgx_linux_x64_driver_1.36.2.bin
sudo chmod a+x sgx_linux_x64_driver_1.36.2.bin
sudo ./sgx_linux_x64_driver_1.36.2.bin
git clone https://github.com/oscarlab/graphene-sgx-driver
cd graphene-sgx-driver
sudo rmmod graphene_sgx || true
sudo rmmod gsgx || true
make
sudo insmod gsgx.ko
ls /dev/gsgx && ls /dev/sgx/enclave && ls /dev/sgx/provision
