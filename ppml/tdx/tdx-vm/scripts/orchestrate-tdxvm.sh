#!/bin/bash

### download guest_image, kernel and tdx.xml
cd ~
mkdir ~/tdx-vm

echo downloading ubuntu guest image: td-guest-ubuntu-22.04.qcow2.tar.xz
wget http://td_guest_url/td-guest-ubuntu-22.04.qcow2.tar.xz -O ~/tdx-vm/td-guest-ubuntu-22.04.qcow2.tar.xz

echo extract td-guest-ubuntu-22.04.qcow2.tar.xz
tar -xvf ~/tdx-vm/td-guest-ubuntu-22.04.qcow2.tar.xz -C ~/tdx-vm

echo downloading guest kernel: vmlinuz-jammy
wget http://td_guest_url/vmlinuz-jammy -O ~/tdx-vm/vmlinuz-jammy

echo downloading tdx.xml
wget http://td_guest_url/tdx.xml -O ~/tdx-vm/tdx.xml

echo destroy vms
virsh destroy tdvm1
virsh destroy tdvm2
virsh destroy tdvm3
virsh undefine tdvm1
virsh undefine tdvm2
virsh undefine tdvm3

rm -rf ~/tdx-vm/vm1
rm -rf ~/tdx-vm/vm2
rm -rf ~/tdx-vm/vm3
mkdir -p ~/tdx-vm/vm1
mkdir -p ~/tdx-vm/vm2
mkdir -p ~/tdx-vm/vm3

cp ~/tdx-vm/td-guest-ubuntu-22.04.qcow2 ~/tdx-vm/vm1/
cp ~/tdx-vm/td-guest-ubuntu-22.04.qcow2 ~/tdx-vm/vm2/
cp ~/tdx-vm/td-guest-ubuntu-22.04.qcow2 ~/tdx-vm/vm3/

cp ~/tdx-vm/vmlinuz-jammy ~/tdx-vm/vm1/
cp ~/tdx-vm/vmlinuz-jammy ~/tdx-vm/vm2/
cp ~/tdx-vm/vmlinuz-jammy ~/tdx-vm/vm3/

cp /usr/share/qemu/OVMF.fd ~/tdx-vm/vm1/
cp /usr/share/qemu/OVMF.fd ~/tdx-vm/vm2/
cp /usr/share/qemu/OVMF.fd ~/tdx-vm/vm3/

cp ~/tdx-vm/tdx.xml ~/tdx-vm/vm1/
cp ~/tdx-vm/tdx.xml ~/tdx-vm/vm2/
cp ~/tdx-vm/tdx.xml ~/tdx-vm/vm3/

echo updating tdx.xml
sed -i "s#<name>td-guest-ubuntu-22.04</name>#<name>tdvm1</name>#g" /root/tdx-vm/vm1/tdx.xml
sed -i "s#<loader>/path/to/OVMF.fd</loader>#<loader>/root/tdx-vm/vm1/OVMF.fd</loader>#g" /root/tdx-vm/vm1/tdx.xml
sed -i "s#<kernel>/path/to/vmlinuz</kernel>#<kernel>/root/tdx-vm/vm1/vmlinuz-jammy</kernel>#g" /root/tdx-vm/vm1/tdx.xml
sed -i "s#<source file='/path/to/td-guest-ubuntu-22.04.qcow2'/>#<source file='/root/tdx-vm/vm1/td-guest-ubuntu-22.04.qcow2'/>#g" /root/tdx-vm/vm1/tdx.xml
sed -i "s#<name>td-guest-ubuntu-22.04</name>#<name>tdvm2</name>#g" /root/tdx-vm/vm2/tdx.xml
sed -i "s#<loader>/path/to/OVMF.fd</loader>#<loader>/root/tdx-vm/vm2/OVMF.fd</loader>#g" /root/tdx-vm/vm2/tdx.xml
sed -i "s#<kernel>/path/to/vmlinuz</kernel>#<kernel>/root/tdx-vm/vm2/vmlinuz-jammy</kernel>#g" /root/tdx-vm/vm2/tdx.xml
sed -i "s#<source file='/path/to/td-guest-ubuntu-22.04.qcow2'/>#<source file='/root/tdx-vm/vm2/td-guest-ubuntu-22.04.qcow2'/>#g" /root/tdx-vm/vm2/tdx.xml
sed -i "s#<name>td-guest-ubuntu-22.04</name>#<name>tdvm3</name>#g" /root/tdx-vm/vm3/tdx.xml
sed -i "s#<loader>/path/to/OVMF.fd</loader>#<loader>/root/tdx-vm/vm3/OVMF.fd</loader>#g" /root/tdx-vm/vm3/tdx.xml
sed -i "s#<kernel>/path/to/vmlinuz</kernel>#<kernel>/root/tdx-vm/vm3/vmlinuz-jammy</kernel>#g" /root/tdx-vm/vm3/tdx.xml
sed -i "s#<source file='/path/to/td-guest-ubuntu-22.04.qcow2'/>#<source file='/root/tdx-vm/vm3/td-guest-ubuntu-22.04.qcow2'/>#g" /root/tdx-vm/vm3/tdx.xml


echo launch vms
virsh define ~/tdx-vm/vm1/tdx.xml
virsh define ~/tdx-vm/vm2/tdx.xml
virsh define ~/tdx-vm/vm3/tdx.xml

virsh start tdvm1
virsh start tdvm2
virsh start tdvm3
