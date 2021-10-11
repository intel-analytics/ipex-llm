Analytics Zoo provides Seq2seq model which is a general-purpose encoder-decoder framework that can be used for Chatbot, Machine Translation and more.

**Highlights**

1. Easy-to-use models, could be fed into NNFrames or BigDL Optimizer for training.
2. Support SimpleRNN, LSTM and GRU.
3. Support transform encoder states before fed into decoder

---
## **Build a Seq2seq model**
You can call the following API in Scala and Python respectively to create a `Seq2seq`.

**Scala**
```scala
val encoder = RNNEncoder[Float](rnnType="lstm", numLayers=3, hiddenSize=3, embedding=Embedding[Float](10, inputSize))
val decoder = RNNDecoder[Float](rnnType="lstm", numLayers=3, hiddenSize=3, embedding=Embedding[Float](10, inputSize))
val bridge = Bridge[Float](bridgeType="dense", decoderHiddenSize=3)
val model = Seq2seq[Float](encoder, decoder, inputShape=SingleShape(List(-1)), outputShape=SingleShape(List(-1)), bridge)
```

* `rnnType`: currently support "simplernn | lstm | gru"
* `numLayer`: number of layers
* `hiddenSize`: hidden size
* `embedding`: embedding layer
* `bridgeType`: currently only support "dense | densenonlinear"
* `input_shape`: shape of encoder input
* `output_shape`: shape of decoder input

**Python**
```python
encoder = RNNEncoder.initialize(rnn_tpye="LSTM", nlayers=1, hidden_size=4)
decoder = RNNDecoder.initialize(rnn_tpye="LSTM", nlayers=1, hidden_size=4)
bridge = Bridge.initialize(bridge_type="dense", decoder_hidden_size=4)
seq2seq = Seq2seq(encoder, decoder, input_shape=[2, 4], output_shape=[2, 4], bridge)
```

* `rnn_type`: currently support "simplernn | lstm | gru"
* `nlayers`: number of layers
* `hidden_size`: hidden size
* `bridge_type`: currently only support "dense | densenonlinear"
* `input_shape`: shape of encoder input
* `output_shape`: shape of decoder input

---
## **Train a Seq2seq model**
After building the model, we can use BigDL Optimizer to train it (with validation) using RDD of [Sample](https://bigdl-project.github.io/master/#APIGuide/Data/#sample).
`feature` is expected to be a sequence(eg. batch x seqLen x feature) and `label` is also a sequence(eg. batch x seqLen x feature).

**Scala**
```scala
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.nn.{TimeDistributedMaskCriterion, ZooClassNLLCriterion}

val optimizer = Optimizer(
model,
trainSet,
TimeDistributedMaskCriterion(
  ZooClassNLLCriterion(paddingValue = padId),
  paddingValue = padId
),
batchSize = 128)

optimizer
  .setOptimMethod(new Adagrad(learningRate = 0.01, learningRateDecay = 0.001))
  .setEndWhen(Trigger.maxEpoch(20))
  .optimize()
```

Also we can use `Seq2seq.fit` api to train the model.
```scala
model.compile(
optimizer = optimMethod,
loss = TimeDistributedMaskCriterion(
  ZooClassNLLCriterion(paddingValue = padId),
  paddingValue = padId
))

model.fit(
  trainSet, batchSize = param.batchSize,
  nbEpoch = 20)
```

**Python**
```python
from bigdl.optim.optimizer import *

optimizer = Optimizer(
    model=seq2seq,
    training_rdd=train_rdd,
    criterion=TimeDistributedMaskCriterion(ZooClassNLLCriterion()),
    end_trigger=MaxEpoch(20),
    batch_size=128,
    optim_method=Adagrad(learningrate=0.01, learningrate_decay=0.001))

optimizer.set_validation(
    batch_size=128,
    trigger=EveryEpoch())
```

Also we can use `Seq2seq.fit` api to train the model.
```python
model.compile(optimizer, loss, metrics)

model.fit(x, batch_size=32, nb_epoch=10, validation_data=None)
```

---
## **Do prediction**
Predict output with given input
**Scala**
```scala
val result = model.infer(input, startSign, maxSeqLen, stopSign, buildOutput)
```
* `input`: a sequence of data feed into encoder, eg: batch x seqLen x featureSize
* `startSign`: a tensor which represents start and is fed into decoder
* `maxSeqLen`: max sequence length for final output
* `stopSign`: a tensor that indicates model should stop infer further if current output is the same with stopSign
* `buildOutput`: Feeding model output to buildOutput to generate final result

**Python**
```python
result = model.infer(input, start_sign, max_seq_len, stop_sign, build_output)
```
* `input`: a sequence of data feed into encoder, eg: batch x seqLen x featureSize
* `start_sign`: a ndarray which represents start and is fed into decoder
* `max_seq_len`: max sequence length for final output
* `stop_sign`: a ndarray that indicates model should stop infer further if current output is the same with stopSign
* `build_output`: Feeding model output to buildOutput to generate final result
---
## **Examples**
We provide an example to train the Seq2seq model on a QA dataset and uses the model to do prediction.

See [here](https://github.com/intel-analytics/analytics-zoo/tree/master/zoo/src/main/scala/com/intel/analytics/zoo/examples/chatbot) for the Scala example.