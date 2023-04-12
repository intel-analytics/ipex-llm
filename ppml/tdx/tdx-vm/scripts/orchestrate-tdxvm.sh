#!/bin/bash

### download guest_image, kernel and tdx.xml
mkdir /root/tdx-vm

echo downloading ubuntu guest image: td-guest-ubuntu-22.04.qcow2.tar.xz
wget http://10.239.45.10:8081/repository/raw/bigdl/tdx-vm/td-guest-ubuntu-22.04.qcow2.tar.xz -O /root/tdx-vm/td-guest-ubuntu-22.04.qcow2.tar.xz

echo extract td-guest-ubuntu-22.04.qcow2.tar.xz
tar -xvf /root/tdx-vm/td-guest-ubuntu-22.04.qcow2.tar.xz -C /root/tdx-vm

echo downloading guest kernel: vmlinuz-jammy
wget http://10.239.45.10:8081/repository/raw/bigdl/tdx-vm/vmlinuz-jammy -O /root/tdx-vm/vmlinuz-jammy

echo downloading tdx.xml
wget http://10.239.45.10:8081/repository/raw/bigdl/tdx-vm/tdx.xml -O /root/tdx-vm/tdx.xml

echo destroy existing vms
for i in {1..$1}; do 
	virsh destroy tdvm$i
	virsh undefine tdvm$i
done

echo prepare guest image, kernel, bios and tdx.xml
for i in {1..$1}; do 
	rm -rf /root/tdx-vm/vm$i
	mkdir -p /root/tdx-vm/vm$i
	cp /root/tdx-vm/td-guest-ubuntu-22.04.qcow2 /root/tdx-vm/vm$i/
	cp /root/tdx-vm/vmlinuz-jammy /root/tdx-vm/vm$i/
	cp /usr/share/qemu/OVMF.fd /root/tdx-vm/vm$i/
	cp /root/tdx-vm/tdx.xml /root/tdx-vm/vm$i/

	sed -i "s#<name>td-guest-ubuntu-22.04</name>#<name>tdvm$i</name>#g" /root/tdx-vm/vm$i/tdx.xml
	sed -i "s#<loader>/path/to/OVMF.fd</loader>#<loader>/root/tdx-vm/vm$i/OVMF.fd</loader>#g" /root/tdx-vm/vm$i/tdx.xml
	sed -i "s#<kernel>/path/to/vmlinuz</kernel>#<kernel>/root/tdx-vm/vm$i/vmlinuz-jammy</kernel>#g" /root/tdx-vm/vm$i/tdx.xml
	sed -i "s#<source file='/path/to/td-guest-ubuntu-22.04.qcow2'/>#<source file='/root/tdx-vm/vm$i/td-guest-ubuntu-22.04.qcow2'/>#g" /root/tdx-vm/vm$i/tdx.xml

done

echo launch vms
for i in {1..$1}; do 
	virsh define /root/tdx-vm/vm$1/tdx.xml
    virsh start tdvm$1
done

