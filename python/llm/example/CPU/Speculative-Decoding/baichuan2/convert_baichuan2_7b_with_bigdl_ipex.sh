set -x
export BAICHUAN2_7B_MODEL_PATH=$BAICHUAN2_7B_MODEL_PATH
mv $BAICHUAN2_7B_MODEL_PATH/modeling_baichuan.py $BAICHUAN2_7B_MODEL_PATH/modeling_baichuan.original.back
mv $BAICHUAN2_7B_MODEL_PATH/tokenization_baichuan.py  $BAICHUAN2_7B_MODEL_PATH/tokenization_baichuan.original.back
cp ./baichuan2_7b_adaptation_for_ipex/modeling_baichuan.bigdl.ipex $BAICHUAN2_7B_MODEL_PATH/modeling_baichuan.py
cp ./baichuan2_7b_adaptation_for_ipex/tokenization_baichuan.bigdl.ipex $BAICHUAN2_7B_MODEL_PATH/tokenization_baichuan.py
