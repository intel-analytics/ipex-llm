#!/bin/bash
set -x
HOST_IP=$(hostname -I | awk '{print $1}')
# local snapshot machine list file is written by self and read by all
SNAPSHOT_MLIST_PATH="/ppml/data/lgbm/$TRAINER_POD_NAME/mlist.snapshot"
# convergenced machine list file is read and written by self
CONVERGENCED_MLIST_PATH="/ppml/data/lgbm/$TRAINER_POD_NAME/mlist.convergenced"
PORT=12400

echo "[INFO] $TRAINER_POD_NAME is trying to work at $HOST_IP:$PORT, restart if crash..."

# write ip:PORT of the trainer to machine list file
if [ -f $SNAPSHOT_MLIST_PATH ]
then
  WRITE_SUCCESS=$(cat $SNAPSHOT_MLIST_PATH|grep "$HOST_IP $PORT")
  if [ ! -z "$WRITE_SUCCESS" ]
  then
    echo "[INFO] $HOST_IP:$PORT has been written to machine list before, fetch machine list of peer trainers..."
  fi
else
  echo "[INFO] writing self socket to both local and convergenced machine list file..."
  echo $HOST_IP $PORT >> $SNAPSHOT_MLIST_PATH
fi

# wait for peer trainer to write their ip:PORT to machine list file
for rank in `seq 0 $(( $TOTAL_TRAINER_COUNT - 1 ))`
do
  PEER_POD_NAME=lgbm-trainer-$rank

  if [ "$PEER_POD_NAME" == "$TRAINER_POD_NAME" ]
  then
    echo $HOST_IP $PORT >> $CONVERGENCED_MLIST_PATH
    continue
  fi

  PEER_TRAINER_MLIST_PATH="/ppml/data/lgbm/$PEER_POD_NAME/mlist.snapshot"
  while [ ! -f  $PEER_TRAINER_MLIST_PATH ]; do
    echo "[INFO] waiting for peer trainer $PEER_POD_NAME to wrtie machine list file..."
    sleep 10
  done

  PEER_TRAINER_SOCKET=$(cat $PEER_TRAINER_MLIST_PATH)
  if [ -f $CONVERGENCED_MLIST_PATH ]
  then
    WRITE_SUCCESS=$(cat $CONVERGENCED_MLIST_PATH|grep "$PEER_TRAINER_SOCKET")
    if [ ! -z "$WRITE_SUCCESS" ]
    then
      echo "[INFO] $PEER_TRAINER_SOCKET has been written to machine list before, fetch machine list of peer trainers..."
    else
      echo $PEER_TRAINER_SOCKET >> $CONVERGENCED_MLIST_PATH
    fi
  else
    echo $PEER_TRAINER_SOCKET >> $CONVERGENCED_MLIST_PATH
  fi
done

echo "[INFO] finished the construction of trainer network topology, start to train..."

# start to train
$LGBM_HOME/lightgbm config=/ppml/data/lgbm/$TRAINER_POD_NAME/train.conf

if [ $? -eq 0 ]
then
  echo "$TRAINER_POD_NAME finished training successfully."
else
  echo "$TRAINER_POD_NAME fails to finish training, its machine list is:"
  cat $CONVERGENCED_MLIST_PATH
fi

rm $CONVERGENCED_MLIST_PATH
