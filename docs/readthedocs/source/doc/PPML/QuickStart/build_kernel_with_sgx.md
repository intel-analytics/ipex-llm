# Building Linux Kernel from Source with SGX Enabled

SGX driver is merged to Linux Kernel from 5.11. After enable SGX feature during kernel building, we don't have to install SGX driver anymore.

In this guide, we show how to build Kernel 5.13 from souce and enable SGX feature on Ubuntu 18.04.


## Prerequisite

Install prerequites for kernel build. Please follow your distro instruction or your favorite way to build kernel.

```
sudo apt-get install flex bison git build-essential kernel-package fakeroot libncurses5-dev libssl-dev ccache

```

## Main steps

Clone Linux Kernel source code.

```
# Obtain Linux kernel source tree
mkdir kernel && cd kernel
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux
# You can change this version
git checkout v5.13
```

Build Kernel from source code with SGX enabled.

```
cp /boot/config-`uname -r` .config
yes '' | make oldconfig
# Enable SGX and SGX KVM
/bin/sed -i 's/^# CONFIG_X86_SGX is not set/CONFIG_X86_SGX=y/g' .config
echo 'CONFIG_X86_SGX_KVM=y' >> .config
make -j `getconf _NPROCESSORS_ONLN` deb-pkg
```

Install kernel from deb and reboot

```
cd ..
sudo dpkg -i linux-headers-5.13.0_5.13.0-1_amd64.deb linux-image-5.13.0_5.13.0-1_amd64.deb
sudo reboot
```

Check if kernel was installed correctly and the SGX driver is working

```bash
$ uname -r
$ ls -l /dev/ | grep sgx
```

## Uninstall this kernel

Uninstall kernel with dpkg (if you want to change back to previous kernel)

```bash
sudo dpkg --purge linux-image-5.13.0 linux-headers-5.13.0
sudo reboot
```

### Trouble Shooting

* Building on Ubuntu 5.4.X may encounter "make[2]: *** No rule to make target 'debian/certs/benh@debian.org.cert.pem', needed by 'certs/x509_certificate_list'.  Stop.". Pls refer to [CONFIG_SYSTEM_TRUSTED_KEYS](https://askubuntu.com/questions/1329538/compiling-the-kernel-5-11-11).
* In some kernels, SGX option is `CONFIG_INTEL_SGX`.
