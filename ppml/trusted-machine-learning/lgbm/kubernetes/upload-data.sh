#!/bin/bash
set -x
for rank in `seq 0 $(( $totalTrainerCount - 1 ))`
do
  TRAINER_POD_NAME=lgbm-trainer-$rank
  TRAINER_DATA_PATH=$nfsMountPath/$TRAINER_POD_NAME
  if [ ! -d $TRAINER_DATA_PATH ]
  then
    mkdir $nfsMountPath/$TRAINER_POD_NAME
  fi
  cp ../binary.train $TRAINER_DATA_PATH
  cp ../binary.test $TRAINER_DATA_PATH
done
