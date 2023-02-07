# Orca NCF Tutorial

## TensorFlow
- Spark DataFrame Users:
    - [Distributed Training and Evaluation](tf_train_spark_dataframe.py)
    - [Distributed Inference](tf_predict_spark_dataframe.py)
    - [Distributed Resume Training](tutorial/NCF/tf_resume_train_spark_dataframe.py)
- Pandas DataFrame Users:
    - [Distributed Training and Evaluation](tf_train_xshards.py)
    - [Distributed Inference](tf_predict_xshards.py)
    - [Distributed Resume Training](tutorial/NCF/tf_resume_train_xshards.py)
- [Transfer Learning](../tf/transfer_learning.py) using Xception with TensorFlow Dataset on cats_vs_dogs dataset.

## PyTorch
- Spark DataFrame Users:
    - [Distributed Training and Evaluation](pytorch_train_spark_dataframe.py)
    - [Distributed Inference](pytorch_predict_spark_dataframe.py)
    - [Distributed Resume Training](pytorch_resume_spark_dataframe.py)
- Pandas DataFrame Users:
    - [Distributed Training and Evaluation](pytorch_train_xshards.py)
    - [Distributed Inference](pytorch_predict_xshards.py)
    - [Distributed Resume Training](pytorch_resume_xshards.py)
- PyTorch DataLoader Users:
    - [Distributed Training and Evaluation](pytorch_train_dataloader.py)
    - [Distributed Resume Training](pytorch_resume_dataloader.py)
- [Transfer Learning](../pytorch/transfer_learning/train.py) using ConvNet with PyTorch DataLoader on hymenoptera dataset.

## Remarks
- Can enable Tensorboard as callback by adding `--tensorboard`
- Can enable learning rate scheduler by adding `--lr_scheduler`
