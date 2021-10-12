Analytics Zoo provides Seq2seq model which is a general-purpose encoder-decoder framework that can be used for Chatbot, Machine Translation and more.
The model could be fed into NNFrames or BigDL Optimizer directly for training.

---
## **Build a Seq2seq Model**
Before build Seq2seq Model, you need build `Encoder`, `Decoder`. And `Bridge` if you want to do some transformation before passing encoder states to decoder.

### **Build an Encoder**
Currently we only support `RNNEncoder` which enables you to put RNN layers into encoder.
You can call the following API in Scala and Python respectively to create a `RNNEncoder`.

**Scala**
```scala
val encoder = RNNEncoder(rnnType, numLayer, hiddenSize, embedding)
```

* `rnnType` style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
* `numLayers` number of layers used in encoder
* `hiddenSize` hidden size of encoder
* `embedding` embedding layer in encoder, default is `null`

You can also define RNN layers yourself
```scala
val encoder = RNNEncoder(rnns, embedding, inputShape)
```

* `rnns` rnn layers used for encoder, support stacked rnn layers
* `embedding` embedding layer in encoder, default is `null`

**Python**
```python
encoder = RNNEncoder.initialize(rnn_type, nlayers, hidden_size, embedding)
```

* `rnn_type` style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
* `nlayers` number of layers used in encoder
* `hidden_size` hidden size of encoder
* `embedding` embedding layer in encoder, default is `None`

Or

```python
encoder = RNNEncoder(rnns, embedding, input_shape)
```

* `rnns` rnn layers used for encoder, support stacked rnn layers
* `embedding` embedding layer in encoder, default is `None`

### **Build a Decoder**
Similar to Encoder, we only support `RNNDecoder` and API is pretty much the same with `RNNEncoder`

**Scala**
```scala
val decoder = RNNDecoder(rnnType, numLayers, hiddenSize, embedding)
```

* `rnnType` style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
* `numLayers` number of layers used in decoder
* `hiddenSize` hidden size of decoder
* `embedding` embedding layer in decoder, default is `null`

You can also define RNN layers yourself
```scala
val decoder = RNNDecoder(rnns, embedding, inputShape)
```

* `rnns` rnn layers used for decoder, support stacked rnn layers
* `embedding` embedding layer in decoder, default is `null`

**Python**
```python
encoder = RNNDecoder.initialize(rnn_type, nlayers, hidden_size, embedding):
```

* `rnn_type` style of recurrent unit, one of [SimpleRNN, LSTM, GRU]
* `nlayers` number of layers used in decoder
* `hidden_size` hidden size of decoder
* `embedding` embedding layer in decoder, default is `None`

Or

```python
decoder = RNNDecoder(rnns, embedding, input_shape)
```

* `rnns` rnn layers used for decoder, support stacked rnn layers
* `embedding` embedding layer in decoder, default is `None`

### **Build a Bridge**
By default, encoder states are directly fed into decoder. In this case, you don't need build a `Bridge`. But if you want to do some transformation before feed encoder states to decoder,
please use following API to create a `Bridge`.

**Scala**
```scala
val bridge = Bridge(bridgeType, decoderHiddenSize)
```

* `bridgeType` currently only support "dense | densenonlinear"
* `decoderHiddenSize` hidden size of decoder

You can also specify various keras layers as a `Bridge`
```scala
val bridge = Bridge(bridge)
```

* `bridge` keras layers used to do the transformation

**Python**
```python
bridge = Bridge.initialize(bridge_type, decoder_hidden_size)
```

* `bridge_type`: currently only support "dense | densenonlinear"
* `decoder_hidden_size`: hidden size of decoder

Or

```python
bridge = Bridge.initialize_from_keras_layer(bridge)
```

* `bridge` keras layers used to do the transformation

### **Build a Seq2seq**

**Scala**
```scala
val seq2seq = Seq2seq(encoder,
    decoder,
    inputShape,
    outputShape,
    bridge,
    generator)
```

* `encoder` an encoder object
* `decoder` a decoder object
* `inputShape` shape of encoder input, for variable length, please input -1
* `outputShape` shape of decoder input, for variable length, please input -1
* `bridge` connect encoder and decoder, you can input `null`
* `generator` Feeding decoder output to generator to generate final result, `null` is supported

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/chatbot) for the Scala example that trains the Seq2seq model and uses the model to do prediction.

**Python**
```python
seq2seq = Seq2seq(encoder, decoder, input_shape, output_shape, bridge,
                 generator)
```

* `encoder` an encoder object
* `decoder` a decoder object
* `input_shape` shape of encoder input, for variable length, please input -1
* `output_shape` shape of decoder input, for variable length, please input -1
* `bridge` connect encoder and decoder, you can input `null`
* `generator` Feeding decoder output to generator to generate final result, `None` is supported
