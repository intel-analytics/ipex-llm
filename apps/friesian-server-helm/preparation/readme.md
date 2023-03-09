# Friesian Online Serving Preparation

This document demonstrates how to create a
resource [persistent volume](https://kubernetes.io/docs/concepts/storage/persistent-volumes/)
for [deploying the Friesian online serving pipeline on a Kubernetes cluster](../friesian-helm/).

## Data preparation

Before installing the Friesian online serving, we need to prepare some resource files.

1. Feature Server

   Users' [WND](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd)
   feature parquet file.

   Items' [WND](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd)
   feature parquet file.

   (Skip this step if having initialized Redis)

2. Feature-Recall Server

   Users' embedding parquet file. (Skip this step if having initialized Redis)

3. Recall Server

   Items' embedding parquet file. (Skip this step if having initialized faiss model)

4. Ranking Server

   [The Wide and Deep Model](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example/wnd)
   trained by recsys data.

## Prepare a resource PV

1. Create a namespace to put Friesian Online Serving. In this demo, we use `friesian`. The following
   steps require this namespace.

    ```bash
    kubectl create ns friesian
    ```

2. Claim a [PV](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) to store resource
   files. Edit `create-pvc.yaml` file as follows:

    * `spec.resources.requests.storage`: Size of the persistent volume, large enough to contain the
      resource files and the faiss model generated during initialization.
    * `spec.storageClassName`: Name of the available
      cluster [storage class](https://kubernetes.io/docs/concepts/storage/storage-classes/). If
      using AWS, Alibaba Cloud, or other cloud services, we can obtain this name in the
      dashboard/console.

    ```bash
    kubectl apply -f 'create-pvc.yaml' -n friesian

    ## After kubectl apply, check if pv created
    kubectl get pv -n friesian
    ```

3. Create a pod to attach the claimed PV.

    ```bash
    kubectl apply -f 'create-volume-pod.yaml' -n friesian
    ```

4. Copy resource files into the attached PV and check files' path.

    ```bash
    # Copy files to the path mounted PV
    kubectl cp /path/to/your/resources friesian/volume-pod:/resources

    # Use pod bash
    kubectl exec -it -n friesian pod/volume-pod -- /bin/bash
    
    # Check the path of the resources relative to /resources
    cd /resources

    # Exit pod bash
    exit

    kubectl delete -f 'create-volume-pod.yaml' -n friesian
    ```

## Resource structure example

```plain
|-resources
  | (Feature Server)
  |-wnd_user.parquet (Users' feature parquet file) 
    |                (Fill wnd_user.parquet in `init.feature.resourcePath.initialUserDataPath`)
    |-part-xxxxxxxxx-xxxx.parquet
    |-part-xxxxxxxxx-xxxx.parquet
    |-...
  |-wnd_item.parquet (Users' feature parquet file) 
    |                (Fill wnd_item.parquet in `init.feature.resourcePath.initialItemDataPath`)
    |-part-xxxxxxxxx-xxxx.parquet
    |-part-xxxxxxxxx-xxxx.parquet
    |-...
  | (Feature-Recall Server)
  |-tmpDir
    |-user_ebd.parquet (Users' embedding parquet file) 
      |                (Fill tmpDir/user_ebd.parquet in `init.featureRecall.resourcePath.initialUserDataPath`)
      |-part-xxxxxxxxx-xxxx.parquet
      |-part-xxxxxxxxx-xxxx.parquet
      |-...
  | (Recall Server)
  |-item_ebd.parquet (Items' embedding parquet file) 
    |                (Fill item_ebd.parquet in `init.recall.resourcePath.initialDataPath`)
    |-part-xxxxxxxxx-xxxx.parquet
    |-part-xxxxxxxxx-xxxx.parquet
    |-...
  | (Ranking Server)
  |-recsys_wnd (The Wide and Deep Model) 
    |          (Fill recsys_wnd in `ranking.resourcePath.modelPath`)
    |-saved_model.pb
    |-assets
      |-...
    |-variables
      |-variables.data-xxxxxxxx
      |-variables.index
      |-...
```
