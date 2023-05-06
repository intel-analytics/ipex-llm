#!/bin/bash

num_of_vms=$1
dir=$2
memory=500
vcpus=180
sockets=2
threads=2
cores=45

### destroy tdx vms if exists

echo destroy existing vms
for ((i=1; i<=$num_of_vms; i++)); do 
  virsh destroy tdvm$i
  virsh undefine tdvm$i --nvram
done

virsh list --all


### download guest_image, kernel and tdx.xml
rm -rf $dir
mkdir -p $dir

echo downloading ubuntu guest image: td-guest-ubuntu-22.04.qcow2.tar.xz
wget https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/guest-images/td-guest-ubuntu-22.04.qcow2.tar.xz -O $dir/td-guest-ubuntu-22.04.qcow2.tar.xz

echo extract td-guest-ubuntu-22.04.qcow2.tar.xz
tar -xvf $dir/td-guest-ubuntu-22.04.qcow2.tar.xz -C $dir

echo downloading guest kernel: vmlinuz-jammy
wget https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/guest-images/vmlinuz-jammy -O $dir/vmlinuz-jammy

echo downloading tdx.xml
wget https://raw.githubusercontent.com/intel/tdx-tools/2022ww44/doc/tdx_libvirt_direct.ubuntu.xml.template -O $dir/tdx.xml

cp /usr/share/qemu/OVMF_VARS.fd $dir/OVMF_VARS.fd

### update tdx.xml

# Set distro related parameters according to distro
DISTRO=$(grep -w 'NAME' /etc/os-release)
if [[ "$DISTRO" =~ .*"Ubuntu".* ]]; then
    QEMU_EXEC="/usr/bin/qemu-system-x86_64"
else
    QEMU_EXEC="/usr/libexec/qemu-kvm"
fi

sed -i "s#<emulator>/usr/bin/qemu-system-x86_64</emulator>#<emulator>$QEMU_EXEC</emulator>#g" $dir/tdx.xml


# Set network
sed -i ':a;N;$!ba;s#<interface type="user">\n.*<model type="virtio"/>\n.*</interface>#placeholder#g' tdx.xml

sed -i 's/placeholder/<interface type="bridge"> \
      <source bridge="virbr0"\/>\
      <model type="virtio"\/>\
      <driver name="vhost"\/>\
    <\/interface>/' tdx.xml


### prepare guest image, kernel, bios and tdx.xml for each VM
echo prepare guest image, kernel, bios and tdx.xml
for ((i=1; i<=$num_of_vms; i++)); do
  rm -rf $dir/vm$i
  mkdir -p $dir/vm$i
  cp $dir/td-guest-ubuntu-22.04.qcow2 $dir/vm$i/
  cp $dir/vmlinuz-jammy $dir/vm$i/
  cp /usr/share/qemu/OVMF_VARS.fd $dir/vm$i/
  cp $dir/tdx.xml $dir/vm$i/

  sed -i "s#<memory unit='KiB'>2097152</memory>#<memory unit='GB'>$memory</memory>#g" $dir/vm$i/tdx.xml
  sed -i "s#<vcpu placement='static'>1</vcpu>#<vcpu placement='static'>$vcpus</vcpu>#g" $dir/vm$i/tdx.xml
  sed -i "s#<topology sockets='1' cores='1' threads='1'/>#<topology sockets='$sockets' cores='$cores' threads='$threads'/>#g" $dir/vm$i/tdx.xml
  sed -i "s#<name>td-guest-ubuntu-22.04</name>#<name>tdvm$i</name>#g" $dir/vm$i/tdx.xml
  sed -i "s#/path/to/OVMF_VARS.fd#$dir/vm$i/OVMF_VARS.fd#g" $dir/vm$i/tdx.xml
  sed -i "s#<kernel>/path/to/vmlinuz</kernel>#<kernel>$dir/vm$i/vmlinuz-jammy</kernel>#g" $dir/vm$i/tdx.xml
  sed -i "s#<source file='/path/to/td-guest-ubuntu-22.04.qcow2'/>#<source file='$dir/vm$i/td-guest-ubuntu-22.04.qcow2'/>#g" $dir/vm$i/tdx.xml

done


### launch tdx vms

echo launch vms
for ((i=1; i<=$num_of_vms; i++)); do
  virsh define $dir/vm$i/tdx.xml
  virsh start tdvm$i
done

virsh list --all
