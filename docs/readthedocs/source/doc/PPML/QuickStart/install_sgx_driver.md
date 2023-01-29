# Install SGX (Software Guard Extensions) Driver for Xeon Server

Checklist for SGX Driver:

1. Please [check if your CPU has SGX feature](https://www.intel.com/content/www/us/en/support/articles/000028173/processors.html).
2. Check if SGX feature is correctly enabled on BIOS. Please ensure enough memory is installed.
   * Disable `UMA-Based Clustering`.
   * Enable `SGX` or `SW Guard Extensions(SGX)`. Set `PRMRR` to the max. Please ensure Reserved Memory Range Registers (PRMRR) are configured for SGX.
   * SGX will reserve some memory from the installed memory. This memory (PRMRR) can not be seen by your system (total memory), e.g., `free -h`. So, `Installed Memory = Total Memory + 2 * PRMRR`.
   * Enable `Auto MP Registration`. This setting is for remote attestation.
3. Recommended OS (Operating System): Ubuntu 18.04/20.04, CentOS 8, Redhat 8.

**Note that SGX driver has been merged to Linux Kernel from 5.11+. After enabling SGX feature during kernel building, SGX driver will be automatically enabled.** So, we recommend our customers upgrade their kernel to 5.14+ with SGX enabled. See [Building Linux Kernel from Source with SGX Enabled](#building-linux-kernel-from-source-with-sgx-enabled).

If your data center cannot upgrade OS or kernel, then you can [Install SGX Driver through the Installation Package](#install-sgx-driver-through-the-installation-package).

## Building Linux Kernel from Source with SGX Enabled

In this guide, we show how to build Kernel 5.14 from the source code and enable SGX feature on Ubuntu 18.04/20.04. You can change the kernel version, i.e., 5.14 if necessary.

### Prerequisite for kernel build

Install prerequisites for kernel build. Please follow your distro instruction or your favorite way to build the kernel.

```bash
sudo apt-get install flex bison git build-essential kernel-package fakeroot libncurses5-dev libssl-dev ccache libelf-dev
```

### Main steps

Clone Linux Kernel source code.

```bash
# Obtain Linux kernel source tree
mkdir kernel && cd kernel
git clone https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git
cd linux
# You can change this version
git checkout v5.14
```

Build Kernel from source code with SGX enabled.

```bash
cp /boot/config-`uname -r` .config
yes '' | make oldconfig
# Enable SGX and SGX KVM
/bin/sed -i 's/^# CONFIG_X86_SGX is not set/CONFIG_X86_SGX=y/g' .config
echo 'CONFIG_X86_SGX_KVM=y' >> .config
make -j `getconf _NPROCESSORS_ONLN` deb-pkg
```

Install kernel from deb and reboot

```bash
cd ..
sudo dpkg -i linux-headers-5.14.0_5.14.0-1_amd64.deb linux-image-5.14.0_5.14.0-1_amd64.deb
sudo reboot
```

Check if Kernel was installed correctly and the SGX driver is working

```bash
uname -r
ls -l /dev/ | grep sgx
```

### Uninstall this kernel

Uninstall kernel with dpkg (if you want to change back to the previous kernel)

```bash
sudo dpkg --purge linux-image-5.14.0 linux-headers-5.14.0
sudo reboot
```

## Install SGX Driver through the Installation Package

**Warning:** This guide is only for customers who cannot enable SGX driver in kernel.

In this guide, we show how to install SGX driver with the installation package. This allows customers to enable SGX without upgrading their OS or kernel. More details in [Intel_SGX_SW_Installation_Guide_for_Linux.pdf](https://download.01.org/intel-sgx/latest/dcap-latest/linux/docs/Intel_SGX_SW_Installation_Guide_for_Linux.pdf).

### Prerequisite for SGX Driver

```bash
sudo apt-get install build-essential ocaml automake autoconf libtool wget python libssl-dev dkms 
```

### Download & Install SGX Driver binary file

```bash
wget - https://download.01.org/intel-sgx/latest/linux-latest/distro/ubuntu20.04-server/sgx_linux_x64_driver_1.41.bin
chmod 777 sgx_linux_x64_driver_1.41.bin
sudo ./sgx_linux_x64_driver_1.41.bin
```

Check if the SGX driver is installed correctly

```bash
ls -l /dev/ | grep sgx
```

If you encounter any issue during installation, please open an issue on [Intel Software Guard Extensions Data Center Attestation Primitives](https://github.com/intel/SGXDataCenterAttestationPrimitives)

## Trouble Shooting

* Building on Ubuntu 5.4.X may encounter
  * "dpkg-source: error: cannot represent change to vmlinux-gdb.py:". Remove `vmlinux-gdb.py`, then build again.
  * "make[2]: *** No rule to make target 'debian/certs/benh@debian.org.cert.pem', needed by 'certs/x509_certificate_list'.  Stop.". Please disable `SYSTEM_TRUSTED_KEYS`, i.e., `CONFIG_SYSTEM_TRUSTED_KEYS=""` in `.config`. Refer to [CONFIG_SYSTEM_TRUSTED_KEYS](https://askubuntu.com/questions/1329538/compiling-the-kernel-5-11-11).
  * "make[4]: *** No rule to make target 'debian/canonical-revoked-certs.pem', needed by 'certs/x509_revocation_list'.  Stop.". Please disable `SYSTEM_REVOCATION_KEYS`, i.e., `CONFIG_SYSTEM_REVOCATION_KEYS=""` in `.config`.
  * "BTF: .tmp_vmlinux.btf: pahole (pahole) is not available. Failed to generate BTF for vmlinux". `dwarves` are missing. `sudo apt-get install dwarves`.
* In some kernels, SGX option is `CONFIG_INTEL_SGX`.
* 5.13 Kernel may encounter nfs problem [Can't mount NFS-shares from Linux-5.13.0](https://forums.gentoo.org/viewtopic-p-8629887.html?sid=f7359b869fb71849d64f3e69bb48503a)
* [Mellanox interface may be disabled on 5.14.0](https://bugzilla.redhat.com/show_bug.cgi?id=2014094). Changes to 5.15.5 will fix this issue.
* Error 404 when downloading binary file. Please go to [intel-sgx-linux](https://download.01.org/intel-sgx/latest/linux-latest/distro) for the latest download link.
