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
tar -xvf td-guest-ubuntu-22.04.qcow2.tar.xz
mkdir tdvm1
cp td-guest-ubuntu-22.04.qcow2 ./tdvm1/
wget https://ubit-artifactory-or.intel.com/artifactory/linuxcloudstacks-or-local/tdx-stack/tdx-2022ww44/guest-images/vmlinuz-jammy
wget https://raw.githubusercontent.com/intel/tdx-tools/2022ww44/doc/tdx_libvirt_direct.ubuntu.xml.template -O tdx1.xml
cp /usr/share/qemu/OVMF_VARS.fd /usr/share/qemu/OVMF_VARS.fd.bk
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
cp /usr/share/qemu/OVMF_VARS.fd ~/tdx-vm/tdvm1/OVMF_VARS.fd.bk
```
edit tdx1.xml
```
replace '<name>td-guest-ubuntu-22.04</name>' with '<name>td-guest-ubuntu-22.04-1</name>'
replace '/path/to/OVMF_VARS.fd' with the desired destination of OVMF_VARS.fd
replace '/path/to/vmlinuz' with the absolute path of '~/vmlinuz-jammy'
replace '/path/to/td-guest-ubuntu-22.04.qcow2' with the absolute path of '~/td-guest-ubuntu-22.04.qcow2'
replace '/usr/bin/qemu-system-x86_64' with the desired destination of qemu-system such as /usr/libexec/qemu-kvm
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
#### 4.3. unset proxy
vi /etc/environment
unset no_proxy
unset NO_PROXY
#### 4.4. install docker
refer to https://docs.docker.com/engine/install/ubuntu/
vim /etc/docker/daemon.json
```
{
  "registry-mirrors": ["https://ustc-edu-cn.mirror.aliyuncs.com"]
}
```
```
systemctl daemon-reload
systemctl restart docker
docker version
```
#### 4.5. install kubeadm
(1) install kubelet, kubectl, and kubeadm. (docker has been deprecated after k8s 1.24.0)
```
apt-get install kubelet=1.23.4-00 kubeadm=1.23.4-00 kubectl=1.23.4-00
```

(2) If the install process is stopped with message like `Unable to locate kubelet`, add the kubernets repository and refresh apt source:
```
echo "deb https://mirrors.aliyun.com/kubernetes/apt/ kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list
or
echo "deb http://mirrors.ustc.edu.cn/kubernetes/apt kubernetes-xenial main" | tee /etc/apt/sources.list.d/kubernetes.list

apt-get update
```

when updating the apt-get source, if you encounter GPG error, fix it as below:

W: GPG error: <https://mirrors.aliyun.com/kubernetes/apt> kubernetes-xenial InRelease: The following signatures couldn't be verified because the public key is not available: NO_PUBKEY FEEA9169307EA071 NO_PUBKEY 8B57C5C2836F4BEB
 
E: The repository '<https://mirrors.aliyun.com/kubernetes/apt> kubernetes-xenial InRelease' is not signed.
 
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
```
curl https://mirrors.aliyun.com/kubernetes/apt/doc/apt-key.gpg |sudo apt-key add -
```
Please replace the variables with the right ones. The key can be found in the outputted GPG error messages.

Run the install command again after fixing the errors. 
```
apt-get install kubelet=1.23.4-00 kubeadm=1.23.4-00 kubectl=1.23.4-00
```

The same steps need to be done on the other nodes.


4. Configure Kubernets Master
Turn to the master-node, use the kubeadm init command to start the cluster master:

```
echo "export KUBECONFIG=/etc/kubernetes/admin.conf" >> /etc/profile
source /etc/profile
echo '{"exec-opts": ["native.cgroupdriver=systemd"]}' | sudo tee /etc/docker/daemon.json
systemctl daemon-reload
systemctl restart docker
systemctl restart kubelet
kubeadm reset
 
sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --ignore-preflight-errors=NumCPU --image-repository=registry.aliyuncs.com/google_containers
```


After a while, you can get the successful initiation message like below:

```
Your Kubernetes control-plane has initialized successfully!
To start using your cluster，you need to run the following as a regular user:
 
  mkdir -p $HOME/ .kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/ .kube/config
  sudo chown $(id -u):$(id -g)$HOME/ .kube/config
 
You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork] .yaml" with one of the options listed at:
  https: //kubernetes.io/docs/concepts/cluster-administration/addons/
kubeadm join 172.168.2.173:6443 --token 4z9x4v.5gr2dhe5oecfuiuv \
        --discovery-token-ca-cert-hash sha256:f59fb04d6e2e0d266ce43b5ea9374b754a2e40af4af23e4f53163af9fec2702d
```

Moreover, note that there will be a kubectl join command with token and hash in the output. Please record it and use that command to join other nodes to the cluster.

Then make the master-node workable:

kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml # if kubectl cannot get the internet file, please wget it first and then kubectl apply the local file
If your kubectl are not accessible to the external network, you can first download the yaml file by wget the link, and then kubectl apply the downloaded file locally.

```
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml # if kubectl cannot get the internet file, please wget it first and then kubectl apply the local file
```

Configure the Nodes
Then run the kubeadm join command, you can find the command in the output of kubeadm init command above:
```
echo '{"exec-opts": ["native.cgroupdriver=systemd"]}' | sudo tee /etc/docker/daemon.json
systemctl daemon-reload
systemctl restart docker
systemctl restart kubelet
kubeadm reset

kubectl join command with token and hash
```

### 5. Run workload
```
kubectl cluster-info # master="k8s://https://127.0.0.1:12345"

kubectl create serviceaccount spark
kubectl create clusterrolebinding spark-role --clusterrole=edit --serviceaccount=default:spark --namespace=default

.kube/config
kubectl config view --flatten --minify > kuberconfig


export K8S_MASTER=k8s://$(kubectl cluster-info | grep 'https.*6443' -o -m 1)
echo The k8s master is $K8S_MASTER .
sudo docker run -itd --net=host \
    -v /etc/kubernetes:/etc/kubernetes \
    -v /root/.kube:/root/.kube \
    -v /root/sparkpi.sh:/root/sparkpi.sh \
    -v /root/test.sh:/root/test.sh \
    -e RUNTIME_SPARK_MASTER=$K8S_MASTER \
    --name bigdl-k8s \
    intelanalytics/bigdl-k8s:2.2.0 bash

${SPARK_HOME}/bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master local[2] \
  --executor-memory 2G \
  --num-executors 2 \
  local://${SPARK_HOME}/examples/jars/spark-examples_2.12-3.1.3.jar \
  1000

${SPARK_HOME}/bin/spark-submit \
  --class org.apache.spark.examples.SparkPi \
  --master ${RUNTIME_SPARK_MASTER} \
  --deploy-mode cluster \
  --executor-memory 500MB \
  --num-executors 2 \
  --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
  --conf spark.kubernetes.executor.deleteOnTermination=false \
  --conf spark.kubernetes.container.image=intelanalytics/bigdl-k8s:2.2.0 \
  local://${SPARK_HOME}/examples/jars/spark-examples_2.12-3.1.3.jar \
  1000

 docker exec -it bigdl-k8s /bin/bash ~/sparkpi.sh

https://blog.csdn.net/weixin_43114954/article/details/119153903
kubectl taint nodes --all node-role.kubernetes.io/master-
```
