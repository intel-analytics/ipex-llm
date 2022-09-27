# Nano Known Issues

## **PyTorch Issues**

### **AttributeError: module 'distutils' has no attribute 'version'**

This usually is because the latest setuptools does not compatible with PyTorch 1.9.

You can downgrade setuptools to 58.0.4 to solve this problem.

For example, if your `setuptools` is installed by conda, you can run:

```bash
conda install setuptools==58.0.4
```

### **error while loading shared libraries: libunwind.so.8**

You may see this error message when running `source bigdl-nano-init`
```
 Sed: error while loading shared libraries: libunwind.so.8: cannot open shared object file: No such file or directory.
```
You can use the following command to fix this issue.

* `apt-get install libunwind8-dev` 

### **Bus error (core dumped) in multi-instance training with spawn distributed backend**

This usually is because you did not set enough shared memory size in your docker container.

You can increase `--shm-size` to a larger value, e.g. a few GB, to your `docker run` command, or use `--ipc=host`.

If you are running in k8s, you can mount larger storage in `/dev/shm`. For example, you can add the following `volume` and `volumeMount` in your pod and container definition.

```yaml
spec:
  containers:
    ...
    volumeMounts:
    - mountPath: /dev/shm
      name: cache-volume
  volumes:
  - emptyDir:
    medium: Memory
    sizeLimit: 8Gi
    name: cache-volume
```

## **TensorFlow Issues**

### **Nano keras multi-instance training currently does not suport tensorflow dataset.from_generators, numpy_function, py_function**

Nano keras multi-instance training will serialize TensorFlow dataset object into a `graph.pb` file, which does not work with `dataset.from_generators`, `dataset.numpy_function`, `dataset.py_function` due to limitations in TensorFlow.

## **Ray Issues**

### **protobuf version error**

Now `pip install ray[default]==1.11.0` will install `google-api-core==2.10.0`, which depends on `protobuf>=3.20.1`. However, nano depends on `protobuf==3.19.4`, so if we install `ray` after installing `bigdl-nano`, pip will reinstall `protobuf==4.21.5`, which causes error.
