# Friesian User Guide

## **Overview**
BigDL Friesian is an open source framework for building and deploying recommender systems and works with the other BigDL components including DLlib and Orca to provide end-to-end recommender solution on Intel CPU. 
It provides: 
1. An offline pipeline, including unified and easy-to-use APIs for [feature engineering](../QuickStart/offline.md), [examples](https://github.com/intel-analytics/BigDL/tree/main/python/friesian/example) of common recommender models.
2. A nearline pipeline of loading procssed data into redis database.
3. An online framework of distributed serving.

## **Install**
Note: For windows Users, we recommend using Windows Subsystem for Linux 2 (WSL2) to run BigDL-Friesian. Please refer [here](./windows_guide.md) for instructions.

BigDL-Friesian can be installed using pip and we recommend installing BigDL-Friesian in a conda environment.

```bash
conda create -n bigdl python==3.7
conda activate bigdl
pip install bigdl-friesian
```

## **Quick Start**

### **Offline feature engineering**
BigDL-Friesian provides `table` APIs for common feature engineering examples.
```python
import pandas as pd
from bigdl.friesian.feature import FeatureTable
movielens_data = movielens.get_id_ratings(data_dir)
pddf = pd.DataFrame(movielens_data, columns=["user", "item", "label"])
num_users, num_items = pddf["user"].max() + 1, pddf["item"].max() + 1
full = FeatureTable.from_pandas(pddf)\
        .apply("label", "label", lambda x: x - 1, 'int')
train, test = full.random_split([0.8, 0.2], seed=1)
```

### Offline distributed training
```python
from bigdl.orca.learn.tf2.estimator import Estimator
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, concatenate, multiply

def build_model(num_users, num_items, class_num, layers=[20, 10]):
    num_layer = len(layers)
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    mlp_embed_user = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2),
                               input_length=1)(user_input)
    mlp_embed_item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2),
                               input_length=1)(item_input)

    user_latent = Flatten()(mlp_embed_user)
    item_latent = Flatten()(mlp_embed_item)

    mlp_latent = concatenate([user_latent, item_latent], axis=1)
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], activation='relu',
                      name='layer%d' % idx)
        mlp_latent = layer(mlp_latent)
    prediction = Dense(class_num, activation='softmax', name='prediction')(mlp_latent)

    model = tf.keras.Model([user_input, item_input], prediction)
    return model
    
    config = {"lr": 1e-3, "inter_op_parallelism": 4, "intra_op_parallelism": executor_cores}
    def model_creator(config):
        model = build_model(num_users, num_items, 5)
        optimizer = tf.keras.optimizers.Adam(config["lr"])
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['sparse_categorical_crossentropy', 'accuracy'])
        return model

    steps_per_epoch = math.ceil(train.size() / batch_size)
    val_steps = math.ceil(test.size() / batch_size)

    estimator = Estimator.from_keras(model_creator=model_creator, config=config)
    estimator.fit(train.df,
                  batch_size=batch_size,
                  epochs=epochs,
                  feature_cols=['user', 'item'],
                  label_cols=['label'],
                  steps_per_epoch=steps_per_epoch,
                  validation_data=test.df,
                  validation_steps=val_steps)
    tf.saved_model.save(estimator.get_model(), args.model_dir)
```



