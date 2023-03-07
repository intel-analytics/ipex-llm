#!/bin/bash
set -x
HOST_IP=$(hostname -I | awk '{print $1}')
MLIST_PATH="/ppml/data/lgbm/mlist.txt"

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
  echo "[INFO] writing machine list file..."
  echo $HOST_IP $port >> $MLIST_PATH
  cat $MLIST_PATH
fi

# wait for peer trainer to write their ip:port to machine list file
while [ $(wc -l < $MLIST_PATH ) != $TOTAL_TRAINER_COUNT ]; do
  echo "[INFO] waiting for peer trainer to wrtie machine list file..."
  cat $MLIST_PATH
  sleep 10
done


# start to train
/ppml/LightGBM/lightgbm config=/ppml/data/lgbm/$TRAINER_POD_NAME/train.conf

# reset the environment for next-time training
if [ -f $MLIST_PATH ]
then
  echo "[INFO] delete machine list file..."
  rm /ppml/data/lgbm/mlist.txt
fi
