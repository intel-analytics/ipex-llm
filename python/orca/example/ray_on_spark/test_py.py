from bigdl.orca import init_orca_context

#cluster_mode = "local"
cluster_mode = "k8s"
if cluster_mode == "local":
    sc = init_orca_context(cluster_mode="local", cores="*", init_ray_on_spark=True)
elif cluster_mode == "k8s":
    sc = init_orca_context(cluster_mode="k8s", cores=8, num_nodes=4, init_ray_on_spark=True,
                           extra_python_lib="/bigdl2.0/data/new.py",
                           master="k8s://https://172.16.0.200:6443",
                           container_image="10.239.45.10/arda/intelanalytics/bigdl-k8s-spark-3.1.2:0.14.0-SNAPSHOT",
                           conf={"spark.driver.host": "172.16.0.200",
                                 "spark.driver.port": "54321",
                                 "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.driver.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.options.claimName": "nfsvolumeclaim",
                                 "spark.kubernetes.executor.volumes.persistentVolumeClaim.nfsvolumeclaim.mount.path": "/bigdl2.0/data"})

def transform(x):
    from new import dummy_func
    #return x + dummy_func()
    import sys
    return sys.path

rdd = sc.parallelize([1, 2, 3, 4, 5])
res = rdd.map(transform).collect()
print(res)


import ray


@ray.remote(num_cpus=1)
class Counter(object):
    def __init__(self):
        self.n = 0

    def increment(self):
        import sys
        print(sys.path)
        sys.path.append(".")
        from new import dummy_func
        self.n += dummy_func()
        return self.n


counters = [Counter.remote() for i in range(5)]
print(ray.get([c.increment.remote() for c in counters]))
