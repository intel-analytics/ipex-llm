# Orca NCF Tutorial

## TensorFlow
- Spark DataFrame Users:
    - [Distributed Training and Evaluation](tf_train_spark_dataframe.py)
    - [Distributed Inference](tf_predict_spark_dataframe.py)
    - [Distributed Resume Training](tf_resume_train_spark_dataframe.py)
- Pandas DataFrame Users (using Orca XShards):
    - [Distributed Training and Evaluation](tf_train_xshards.py)
    - [Distributed Inference](tf_predict_xshards.py)
    - [Distributed Resume Training](tf_resume_train_xshards.py)

## PyTorch
- Spark DataFrame Users:
    - [Distributed Training and Evaluation](pytorch_train_spark_dataframe.py)
    - [Distributed Inference](pytorch_predict_spark_dataframe.py)
    - [Distributed Resume Training](pytorch_resume_train_spark_dataframe.py)
- Pandas DataFrame Users (using Orca XShards):
    - [Distributed Training and Evaluation](pytorch_train_xshards.py)
    - [Distributed Inference](pytorch_predict_xshards.py)
    - [Distributed Resume Training](pytorch_resume_train_xshards.py)
- PyTorch DataLoader Users:
    - [Distributed Training and Evaluation](pytorch_train_dataloader.py)
    - [Distributed Resume Training](pytorch_resume_train_dataloader.py)

## Remarks
- Can enable Tensorboard as callback by adding `--tensorboard`
- Can enable learning rate scheduler by adding `--lr_scheduler`
