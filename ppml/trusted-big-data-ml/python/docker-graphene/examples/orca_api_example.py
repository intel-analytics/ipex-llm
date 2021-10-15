from zoo.orca import init_orca_context, stop_orca_context
from tensorflow import keras
from zoo.pipeline.api.keras.layers import *
import argparse
import tensorflow as tf
import os

def bigdl_estimator():
    from zoo.orca.learn.bigdl.estimator import Estimator
    from tensorflow.python.keras.datasets import imdb
    from tensorflow.python.keras.preprocessing import sequence
    from zoo.pipeline.api.keras.models import Model
    from zoo.pipeline.api.keras.objectives import SparseCategoricalCrossEntropy
    from zoo.orca.data import XShards
    from zoo.orca.learn.metrics import Accuracy
    import numpy as np

    # conf = {"spark.executor.extraJavaOptions": "-Xss512m", "spark.driver.extraJavaOptions": "-Xss512m"}

    # init_orca_context(cluster_mode="local", cores=8, memory="16g")
    init_orca_context(cluster_mode="local", cores=4, memory="16g")
    max_features = 200
    max_len = 20

    print("running bigdl estimator")

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[-1000:]
    y_test = y_test[-1000:]
    
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    train_pos = np.zeros((len(x_train), max_len), dtype=np.int32)
    val_pos = np.zeros((len(x_test), max_len), dtype=np.int32)
    for i in range(0, len(x_train)):
        train_pos[i, :] = np.arange(max_len)
        val_pos[i, :] = np.arange(max_len)

    train_dataset = XShards.partition({"x": (x_train, train_pos), "y": np.array(y_train)})
    val_dataset = XShards.partition({"x": (x_test, val_pos), "y": np.array(y_test)})

    token_shape = (max_len,)
    position_shape = (max_len,)
    token_input = Input(shape=token_shape)
    position_input = Input(shape=position_shape)
    O_seq = TransformerLayer.init(vocab=max_features, hidden_size=128, n_head=8, seq_len=max_len)([token_input, position_input])
    # Select the first output of the Transformer. The second is the pooled output.
    O_seq = SelectTable(0)(O_seq)
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.2)(O_seq)
    outputs = Dense(2, activation='softmax')(O_seq)

    model = Model([token_input, position_input], outputs)
    model.summary()
    batch_size = 64
    print("Train started")
    est = Estimator.from_bigdl(model=model, loss=SparseCategoricalCrossEntropy(), optimizer=Adam(), metrics=[Accuracy()])
    est.set_constant_gradient_clipping(0.1, 0.2)
    est.fit(data=train_dataset, batch_size=batch_size, epochs=1)
    result = est.evaluate(val_dataset)
    print(result)
    est.clear_gradient_clipping()
    est.set_l2_norm_gradient_clipping(0.5)
    est.fit(data=train_dataset, batch_size=batch_size, epochs=1)
    print("Train finished") 
    
    print("Evaluating started")
    result = est.evaluate(val_dataset)
    print(result)
    print("Evaluating finished")
    est.save('work/saved_model')
    # est.load('work/saved_model')
    print("load and save API finished")

    est.get_train_summary(tag='Loss')
    est.get_validation_summary(tag='Top1Accuracy')
    print("get summary API finished")


    stop_orca_context()


def tf_estimator():
    from zoo.orca.learn.tf.estimator import Estimator
    init_orca_context(cluster_mode="local", cores=4, memory="3g")
    
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    print("running tf estimator")
    
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
    # print(train_data)
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3
    
    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', 
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post',
                                                            maxlen=256)
    
    model = keras.Sequential()
    model.add(keras.layers.Embedding(1000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()
    
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])

    x_val = train_data[:1000]
    partial_x_train = train_data[1000:]

    y_val = train_labels[:1000]
    partial_y_train = train_labels[1000:]

    train_dataset = tf.data.Dataset.from_tensor_slices((partial_x_train, partial_y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    est = Estimator.from_keras(keras_model=model)
    est.set_constant_gradient_clipping(0.1, 0.2)
    est.fit(data=train_dataset, batch_size=512, epochs=5, validation_data=validation_dataset)
    results = est.evaluate(validation_dataset)
    print(results)
    est.clear_gradient_clipping()
    est.set_l2_norm_gradient_clipping(0.1)
    est.fit(data=train_dataset, batch_size=512, epochs=5, validation_data=validation_dataset)
    results = est.evaluate(validation_dataset)
    print(results)
    est.save('work/saved_model')
    print("save API finished")
    # est.save_tf_checkpoint('work/checkpoint')
    # est.load_tf_checkpoint('work/checkpoint')
    print("checkpoint save and load API finished")
    est.save_keras_model('work/keras_model')
    est.save_keras_weights('work/keras_weights')
    print("keras model and weights save API finished")
    # est.load_keras_model('work/keras_model')
    # est.load_keras_weights('work')
    print("keras model and weights load API finished")
    est.get_train_summary(tag='Loss')
    est.get_validation_summary(tag='Top1Accuracy')
    # Estimator.load(est, model_path='work/') # Has not been implemented
    # resutls = est.predict(validation_dataset)
    # print(results)

    stop_orca_context()

def tf2_estimator():
    
    from zoo.orca.learn.tf2.estimator import Estimator
    # import ray
    init_orca_context(cluster_mode="local", cores=4, memory="3g")

    print("running tf2 estimator")
    
    imdb = keras.datasets.imdb
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=1000)
    # print(train_data)
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post',
                                                            maxlen=256)

    test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post',
                                                            maxlen=256)

    model = keras.Sequential()
    model.add(keras.layers.Embedding(1000, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

    model.summary()

    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['acc'])
    
    x_val = train_data[:1000]
    partial_x_train = train_data[1000:]

    y_val = train_labels[:1000]
    partial_y_train = train_labels[1000:]

    train_dataset = tf.data.Dataset.from_tensor_slices((partial_x_train, partial_y_train))
    validation_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    est = Estimator.from_keras(model_creator=model)
    est.fit(data=train_dataset, batch_size=512, epochs=100, validation_data=validation_dataset)
    results = est.evaluate(validation_dataset)
    print(results)
    est.save('work/saved_model')
    est.get_train_summary(tag='Loss')
    est.get_validation_summary(tag='Top1Accuracy')


    stop_orca_context()


def pytorch_estimator():
    
    print("running pytorch estimator")
    
    return

def openvino_estimator():
    
    print("running openvino estimator")

    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="bigdl",
                    help='The mode for the examples. bigdl, tf, tf2, pytorch or openvino.')
    args = parser.parse_args()
    mode = args.mode
    if mode == "bigdl":
        bigdl_estimator()
    elif mode == "tf":
        tf_estimator()
    elif mode == "tf2":
        tf2_estimator()
    elif mode == "pytorch":
        pytorch_estimator()
    else:
        openvino_estimator()


