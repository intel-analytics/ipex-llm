Run Simple Query on TDX-VM

### 1. Hardware Configuration and BIOS Configuration
refer to [Getting_Started_External.pdf (2022ww44)](https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/Getting_Started_External.pdf)

### 2. Build host and guest packages
refer to [Getting_Started_External.pdf (2022ww44)](https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/Getting_Started_External.pdf)

### 3. Launch Ubuntu TDX VM via Libvirt
#### 3.1. Download ubuntu image and kernel
```
cd ~
mkdir tdx-vm && cd tdx-vm
wget https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/guest-images/td-guest-ubuntu-22.04.qcow2.tar.xz
wget https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/guest-images/vmlinuz-jammy
wget https://raw.githubusercontent.com/intel/tdx-tools/2022ww44/doc/tdx_libvirt_direct.ubuntu.xml.template -O tdx1.xml
```


#### 3.2. Configure libvirt
As the root user, uncomment and save the following settings in /etc/libvirt/qemu.conf:
```
user = "root"
group = "root"
dynamic_ownership = 0
```
To make sure libvirt uses these settings, restart the libvirt service:
```
sudo systemctl restart libvirtd
```
#### 3.3. Edit the configuration file
```
cp OVMF_VARS.fd OVMF_VARS.fd.bk
```
edit tdx1.xml
```
replace '<name>td-guest-ubuntu-22.04</name>' with '<name>td-guest-ubuntu-22.04-1</name>'
replace '/path/to/OVMF_VARS.fd' with the desired destination of OVMF_VARS.fd
replace '/path/to/vmlinuz' with the absolute path of '~/vmlinuz-jammy'
replace '/path/to/td-guest-ubuntu-22.04.qcow2' with the absolute path of '~/td-guest-ubuntu-22.04.qcow2'
replace '/usr/bin/qemu-system-x86_64' with the desired destination of qemu-system-x86_64
replace '<vcpu placement='static'>1</vcpu>' with '<vcpu placement='static'>8</vcpu>'
replace '<topology sockets='1' cores='1' threads='1'/>' with '<topology sockets='1' cores='8' threads='1'/>'
```
```
cp tdx1.xml tdx2.xml
```
edit tdx2.xml
```
replace '<name>td-guest-ubuntu-22.04-1</name>' with '<name>td-guest-ubuntu-22.04-2</name>'
replace '/path/to/OVMF_VARS.fd' with the desired destination of OVMF_VARS.fd
replace '/path/to/td-guest-ubuntu-22.04.qcow2' with the absolute path of '~/td-guest-ubuntu-22.04.qcow2'
```
#### 3.4. Configure network between TDX VMs
##### NAT mode
in tdx*.xml, replace
```
    <interface type="user">
      <model type="virtio"/>
    </interface>
```
with 
```
    <interface type="bridge">
      <source bridge="virbr0"/>
      <model type="virtio"/>
      <driver name="vhost"/>
    </interface>
```
##### Bridge mode
refer to https://www.tecmint.com/create-network-bridge-in-rhel-centos-8/ to setup bridge notwork mode on host. Then in tdx*.xml, replace
```
    <interface type="user">
      <model type="virtio"/>
    </interface>
```
with 
```
    <interface type="bridge">
      <source bridge="br0"/>
      <model type="virtio"/>
    </interface>
```
#### 3.5. Launch TDM VM
```
sudo virsh define tdx1.xml
sudo virsh start td-guest-ubuntu-22.04-1

sudo virsh define tdx2.xml
sudo virsh start td-guest-ubuntu-22.04-2
```
Once the TD guest VM is launched, you can verify it is truly TDX VM by querying cpuinfo:
```
cat /proc/cpuinfo | grep tdx_guest
flags : fpu vme de pse tsc msr pae cx8 apic sep pge mca cmov pat pse36 clflush dts mmx fxsr sse sse2 ss ht syscall
nx pdpe1gb rdtscp lm constant_tsc bts rep_good nopl xtopology tsc_reliable cpuid tsc_known_freq pni pclmulqdq dtes64 ds_cpl
ssse3 sdbg fma cx16 pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand hypervisor
lahf_lm abm 3dnowprefetch cpuid_fault invpcid_single ssbd ibrs ibpb stibp ibrs_enhanced tdx_guest fsgsbase bmi1 hle
avx2 smep bmi2 erms invpcid rtm avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw
avx512vlxsaveopt xsavec xgetbv1 xsaves avx_vnni avx512_bf16 wbnoinvd arat avx512vbmi umip pku ospke waitpkg avx512_vbmi2
shstk gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq la57 rdpid bus_lock_detect cldemote movdir
```
you can ping another TDX VM's ip to see if the network works.

### 4. Setup K8S cluster
#### 4.1. update hostname
#### 4.2. edit /etc/hosts
#### 4.3. 






