# Configure the variables to be passed into the templates.
set -x
export imageName=intelanalytics/bigdl-ppml-trusted-machine-learning-gramine-reference:2.3.0-SNAPSHOT
export totalTrainerCount=2
export trainerPort=12400
export nfsMountPath=a_host_path_mounted_by_nfs_to_upload_data_before_training

bash upload-data.sh
envsubst < lgbm-trainer.yaml | kubectl apply -f -
