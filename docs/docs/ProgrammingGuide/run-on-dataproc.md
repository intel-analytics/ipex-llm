## **Deploy Analytics Zoo with BigDL on Dataproc**

Before using Analytics Zoo and BigDL on Google Dataproc, you need setup a project and create a cluster on Dataproc 
(you may refer to [https://cloud.google.com/sdk/docs/how-to](https://cloud.google.com/dataproc/docs/how-to) 
for more instructions). 
Now you can create a Cloud Dataproc cluster using the Google Cloud SDK's (https://cloud.google.com/sdk/docs/) 
`gcloud` command-line tool.

Note:
 The actual version of the initialization script with Zoo support is still under review here: [https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/pull/469]).
 So at the time of writing you should download and place the updated version of this script somewhere accessible for you 
 (into your own Google Storage bucket, for example) and set appropriate location for `--initialization-actions`.
 
  
You can use use this initialization action to create a new Dataproc cluster with Analytics Zoo and BigDL pre-installed 
in it.

By default, it will automatically download only BigDL 0.7.2 for Dataproc 1.3 (Spark 2.3 and Scala 2.11.8).
So you must specify to download Analytics Zoo instead (which includes BigDL) with: `bigdl-download-url` 
property in metadata:

```bash
gcloud dataproc clusters create <CLUSTER_NAME> \
    --image-version 1.3 \
    --initialization-actions gs://dataproc-initialization-actions/bigdl/bigdl.sh \
    --initialization-action-timeout 10m \
    --metadata 'bigdl-download-url=https://repo1.maven.org/maven2/com/intel/analytics/zoo/analytics-zoo-bigdl_0.7.2-spark_2.3.1/0.4.0/analytics-zoo-bigdl_0.7.2-spark_2.3.1-0.4.0-dist-all.zip'
```

To download a different version of Zoo or one targeted to a different version of Spark/Scala, 
find the download URL from the [Analytics Zoo releases page](https://analytics-zoo.github.io/0.4.0/#release-download/) 
or [maven repository](https://repo1.maven.org/maven2/com/intel/analytics/zoo/), 
and set the metadata key "bigdl-download-url" 
.

More information please refer https://github.com/GoogleCloudPlatform/dataproc-initialization-actions/tree/master/bigdl

Once the cluster is provisioned, you will be able to see the cluster running in the Google Cloud Platform Console. Now you can SSH to the master node.

Cloud Dataproc support various way to SSH to the master, here we use SSH from Google Cloud SDK.
E.g.,
```bash
gcloud compute --project <PROJECT_ID> ssh --zone <ZONE> <CLUSTER_NAME>
```
Google cloud SDK will perform the authentication for you and open an SSH client (Eg Putty).

You should be able to find Analytics Zoo and BigDL located under /opt/intel-bigdl. 
Now you can run jobs with Zoo and BigDL on Google Dataproc 
as usual with `gcloud dataproc jobs submit spark`.
