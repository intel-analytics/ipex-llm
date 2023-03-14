# Configure the variables to be passed into the templates.
set -x
export imageName=intelanalytics/bigdl-ppml-trusted-machine-learning-gramine-reference:2.3.0-SNAPSHOT
export totalTrainerCount=2
export trainerPort=12400

envsubst < lgbm-trainer.yaml | kubectl apply -f -
