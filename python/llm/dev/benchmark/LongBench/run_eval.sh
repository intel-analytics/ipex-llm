config=( 256 512 1024 2048 )

for l in "${config[@]}"; do
    echo dataset: $ds, config: $l
    python eval.py --model mistral-7B-instruct-v0.2ablation_c${l}_w32_k7_maxpool
done

python eval.py --model mistral-7B-instruct-v0.2
