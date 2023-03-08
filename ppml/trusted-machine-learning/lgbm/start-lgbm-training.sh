#!/bin/bash
set -x
HOST_IP=$(hostname -I | awk '{print $1}')
MLIST_PATH="/ppml/data/lgbm/$TRAINER_POD_NAME/mlist.txt"

# find a free port to avoid conflict if multiple trainers are assigned to the same machine
BASE_PORT=12000
INCREMENT=1

port=$BASE_PORT
isfree=$(netstat -taln | grep $port)

while [[ -n "$isfree" ]]; do
  port=$[port+INCREMENT]
  isfree=$(netstat -taln | grep $port)
done

echo "[INFO] $TRAINER_POD_NAME is trying to work at $HOST_IP:$port, restart if crash..."

# write ip:port of the trainer to machine list file
if [ -f $MLIST_PATH ] && [ grep -xqFe "$HOST_IP $port" $MLIST_PATH ]
then
  echo "[INFO] $HOST_IP:$port has been written to machine list before, start to train..."
else
  echo "[INFO] writing local machine list file..."
  echo $HOST_IP $port >> $MLIST_PATH
  cat $MLIST_PATH
fi

# wait for peer trainer to write their ip:port to machine list file
for rank in {0..$TOTAL_TRAINER_COUNT}
do
  PEER_POD_NAME=lgbm-trainer-$rank

  if [ "$PEER_POD_NAME" == "$TRAINER_POD_NAME" ]
  then
    continue
  fi

  PEER_TRAINER_MLIST_PATH="/ppml/data/lgbm/$PEER_POD_NAME/mlist.txt"
  while [ ! -f  $PEER_TRAINER_MLIST_PATH ]; do
    echo "[INFO] waiting for peer trainer $PEER_POD_NAME to wrtie machine list file..."
    sleep 10
  done

  PEER_TRAINER_SOCKET=$(cat $PEER_TRAINER_MLIST_PATH)
  echo $PEER_TRAINER_SOCKET >> $MLIST_PATH
done

echo "[INFO] finished the construction of trainer network topology, start to train..."
cat $MLIST_PATH

# start to train
/ppml/LightGBM/lightgbm config=/ppml/data/lgbm/$TRAINER_POD_NAME/train.conf
