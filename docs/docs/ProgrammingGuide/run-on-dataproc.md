---

## **The Google Cloud Dataproc Initialization Script**

To make it easier to try out BigDL examples on Spark using Google Cloud Dataproc, a public initialization script is provided (the source script is also avaliable in this repo path `scripts/launch-dataproc.sh`). The script will automatically retrieve BigDL package (version 0.2.0), run it on Dataproc's Spark Yarn cluster, then configure and setup the Jupyter Notebook and Tensorboard for the interactive usage. Two examples, including LeNet and Text Classifier, will be provided in the Notebook.

---
## **Before You Start**

Before using BigDL on Dataproc, you need a valid Google Cloud account and setup your Google Cloud SDK (you may refer to [https://cloud.google.com/sdk/docs/how-to](https://cloud.google.com/sdk/docs/how-to) for more instructions).

---
## **Create Spark Cluster with BigDL**

Run the following command to create your cluster
```bash
gcloud dataproc clusters create bigdl \
    --initialization-actions gs://dataproc-initial/bigdl.sh \
    --worker-machine-type n1-highmem-4 \
    --master-machine-type n1-highmem-2 \
    --num-workers 2 \
    --zone us-central1-b \
    --image-version 1.1
```
You can change `bigdl` into any other name as the cluster name, and you are also free to upload `scripts/launch-dataproc.sh` into your own Google Cloud Storage bucket and use it instead of `gs://dataproc-initial/bigdl.sh` in the initialization-actions field.

When creating a larger cluster with more workers, it is suggested to pass the number of executor into the script via the metadata field as, 
```bash
gcloud dataproc clusters create bigdl \
    --initialization-actions gs://dataproc-initial/bigdl.sh \
    --metadata "NUM_EXECUTORS=8" \
    --worker-machine-type n1-highmem-4 \
    --master-machine-type n1-highmem-2 \
    --num-workers 4 \
    --num-preemptible-workers 4 \
    --zone us-central1-b \
    --image-version 1.1
```

Please note that it is highly recommended to run BigDL in the region where the compute instances come with Xeon E5 v3 or v4 processors (you may find the [Google Cloud Regions and Zones](https://cloud.google.com/compute/docs/regions-zones/regions-zones) for more details).

---
## **Play Around with BigDL**
Once your dataproc cluster is ready, directly go to the following URL (change `bigdl` into your own cluster name if you are using a different one) to play around BigDL in Jupyter Notebook. Note that you need to [create an SSH tunel and SOCKS proxy](https://cloud.google.com/dataproc/docs/concepts/cluster-web-interfaces) to visit them. 

* Jupyter Notebook: [http://bigdl-m:8888/](http://bigdl-m:8888/)

* Tensorboard: [http://bigdl-m:6006/](http://bigdl-m:6006/)

* YARN ResourceManager: [http://bigdl-m:8088/](http://bigdl-m:8088/)

Inside your Jupyter Notebook, you may find two examples are already there. Start your BigDL journey with them.
