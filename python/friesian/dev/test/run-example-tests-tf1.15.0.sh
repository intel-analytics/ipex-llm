export FTP_URI=$FTP_URI


set -e
mkdir -p result
echo "#1 start example test for dien train"
#timer
start=$(date "+%s")
if [ -d data/input_dien_train ]; then
  echo "data/input_dien_train already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/input_dien_train.tar.gz -P data
  tar -xvzf data/input_dien_train.tar.gz -C data
fi

python ../../example/dien/dien_train.py \
    --executor_cores 4 \
    --executor_memory 50g \
    --batch_size 128 \
    --data_dir ./data/input_dien_train \
    --model_dir ./result

now=$(date "+%s")
time1=$((now - start))

rm -rf data
rm -rf result

echo "#1 dien train time used: $time1 seconds"
