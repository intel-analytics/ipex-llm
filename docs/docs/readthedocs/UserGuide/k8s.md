### **1. Pull `hyper-zoo` Docker Image**

You may pull the prebuilt  Analytics Zoo `hyper-zoo` Image from [Docker Hub](https://hub.docker.com/r/intelanalytics/hyper-zoo/tags) as follows:

```bash
sudo docker pull intelanalytics/hyper-zoo:latest
```

**Speed up pulling image by adding mirrors**

To speed up pulling the image from DockerHub, you may add the registry-mirrors key and value by editing `daemon.json` (located in `/etc/docker/` folder on Linux):
```
{
  "registry-mirrors": ["https://<my-docker-mirror-host>"]
}
```
For instance, users in China may add the USTC mirror as follows:
```
{
  "registry-mirrors": ["https://docker.mirrors.ustc.edu.cn"]
}
```

After that, flush changes and restart docker：

```
sudo systemctl daemon-reload
sudo systemctl restart docker
```

### **2. Launch a K8s Client Container**

### **3. Run Analytics Zoo Examples on k8s**

_**Note**: Please make sure `kubectl` has appropriate permission to create, list and delete pod._

#### **3.1 Run Python programs**

#### **3.2 Run Jupyter Notebooks**

#### **3.3 Run Scala programs**

#### **3.4 Access logs and clear pods**
