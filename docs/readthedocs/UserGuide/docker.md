# Docker User Guide

---

### **1. Pull Docker Image**

You may pull a Docker image from the  [Analytics Zoo Docker Hub](https://hub.docker.com/r/intelanalytics/analytics-zoo).

To pull the latest version, use <TODO: is this nightly build>
```bash
sudo docker pull intelanalytics/analytics-zoo:latest
```

**Configuring resources**

For Docker Desktop users, the default resources (2 CPUs and 2GB memory) are relatively small, and you may want to change them to larger values (8GB memory and 4 CPUs should be a good estimator for most examples, and the exact memory requirements vary for different applications). For more information, view the Docker documentation for [MacOS](https://docs.docker.com/docker-for-mac/#resources) and [Windows](https://docs.docker.com/docker-for-windows/#resources).

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

### **2. Launch Docker Container**

### **3. Run Jupyter Notebook Examples in the Container**

#### **3.1 Start the Jupyter Notebook services**

#### **3.2 Connect to Jupyter Notebook service from a browser**

#### **3.3 Run Analytics Zoo Jupyter Notebooks**

### **4. Shut Down Docker Container**
