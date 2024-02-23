## C-Eval Benchmark Test

C-Eval benchmark test allows users to test on [C-Eval](https://cevalbenchmark.com) datasets, which is a multi-level multi-discipline chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels. Please check [paper](https://arxiv.org/abs/2305.08322) and [github repo](https://github.com/hkust-nlp/ceval) for more information.

### Download dataset
Please download and unzip the dataset for evaluation.
```shell
wget https://huggingface.co/datasets/ceval/ceval-exam/resolve/main/ceval-exam.zip
mkdir data
mv ceval-exam.zip data
cd data; unzip ceval-exam.zip
```

### Run
You can run evaluation with following command.
```shell
bash run.sh
```
+ `run.sh`
```shell
python eval.py \
    --model_path "path to model" \
    --eval_type validation \
    --device xpu \
    --eval_data_path data \
    --qtype sym_int4
```

> **Note**
>
> `eval_type` there is two types of evaluation, first type is `validation`, which runs on validation dataset and output evaluation scores. The second type is `test`, which runs on test dataset and output `submission.json` file for submission on https://cevalbenchmark.com to get the evaluation score.
