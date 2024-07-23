export HF_ENDPOINT=https://hf-mirror.com

# conda activate yina-kv

# datasets=( "multi_news" "triviaqa" "2wikimqa" "lcc" )
# datasets=( "qasper" )
datasets=( "multi_news" )

config=( 256 512 1024 2048 )

for ds in "${datasets[@]}"; do
    echo dataset: $ds....
    # for l in "${config[@]}"; do
    #     echo dataset: $ds, config: $l
    #     CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.2 --dataset $ds --compress_args_path ablation_c${l}_w32_k7_maxpool.json
    # done
    echo dataset: $ds, original
    CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.2 --dataset $ds
done