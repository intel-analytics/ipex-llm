/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.reflect.ClassTag

/**
 * Transformer model from "Attention Is All You Need".
 * The Transformer model consists of an encoder and a decoder. Both are stacks
 * of self-attention layers followed by feed-forward layers. This model yields
 * good results on a number of problems, especially in NLP and machine translation.
 * See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
 * description of the model and the results obtained with its early version.
 * @param hiddenSize
 * @param numHeads
 * @param filterSize
 * @param numHiddenlayers
 * @param postprocessDropout
 * @param attentionDropout
 * @param reluDropout
 * @tparam T The numeric type in this module parameters.
 */
class TransformerLayer[T: ClassTag](
   val vocabSize: Int,
   val hiddenSize: Int,
   val numHeads: Int,
   val filterSize: Int,
   val numHiddenlayers: Int,
   val postprocessDropout: Float,
   val attentionDropout: Float,
   val reluDropout: Float,
   val problem: ProblemType = LanguageModel)
  (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  require(problem == LanguageModel, "Transformer layer only support language model now")

  override def buildModel(): Module[T] = {
    val input = Input()
    val embedding = LookupTable[T](nIndex = vocabSize, nOutput = hiddenSize).inputs(input)
    val decoder_input = new TransformerPrepareDecoder().inputs(embedding)
    val decoder_self_attention_bias = new TransformerConstant().inputs(embedding)

    val decoder_input_drop = if (train) {
      val postDropOut = Dropout(1- postprocessDropout)
      postDropOut.inputs(decoder_input)
    } else decoder_input

    val blockModel = decode(numHiddenlayers)
    val output = blockModel.inputs(decoder_input_drop, decoder_self_attention_bias)
    Graph(input, output)
  }

  private[nn] def decode(num_layers: Int): Module[T] = {
    val decoder_input = Input()
    val decoder_self_attention_bias = Input()
    var output = decoder_input
    var i = 0
    while (i < num_layers) {
      val attention = new Attention[T](hiddenSize, numHeads, attentionDropout)
      val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
      // for self attention
      val attentionModel = prePostProcessingSelfAttention(
        attention, output, decoder_self_attention_bias, s"self_attention_${i}")
      val ffnModel = prePostProcessingFFN(ffn, attentionModel, s"ffn_${i}")
      output = ffnModel
      i += 1
    }
    val norm = new LayerNormalization[T](hiddenSize).setName("norm").inputs(output)
    Graph(Array(decoder_input, decoder_self_attention_bias), Array(norm))
  }

  private def prePostProcessingSelfAttention(layer: Module[T], decoder_input: ModuleNode[T],
    decoder_self_attention_bias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
        .inputs(decoder_input)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
        .inputs(layer.setName(preName + "/self_attention")
        .inputs(norm, norm, decoder_self_attention_bias))
    CAddTable().inputs(decoder_input, drop)
  }
  private def prePostProcessingFFN(layer: Module[T],
    decoder_input: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoder_input)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/ffn").inputs(norm))
    CAddTable().inputs(decoder_input, drop)
  }
}


object TransformerLayer {
  def apply[T: ClassTag](
     vocabSize: Int,
     hiddenSize: Int,
     numHeads: Int,
     filterSize: Int,
     numHiddenlayers: Int,
     postprocessDropout: Float,
     attentionDropout: Float,
     reluDropout: Float)
   (implicit ev: TensorNumeric[T]): TransformerLayer[T] =
    new TransformerLayer(vocabSize, hiddenSize, numHeads,
      filterSize, numHiddenlayers,
      postprocessDropout, attentionDropout, reluDropout)
}

private[nn] class TransformerConstant[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  /**
    * Create an bias tensor to be added to attention logits.
    * Returns tensor with shape (1, 1, length, length)
    * @param length
    * @tparam T
    * @return
    */
  private def attentionBiasLowerTriangle[T: ClassTag](
    length: Int, output: Tensor[T])(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val arr = output.storage().array()
    for (i <- 0 to (length - 1)) {
      var j = length - 1
      while (j > i) {
        arr(i * length + j) = ev.fromType(-1e9)
        j -= 1
      }
    }
    output.resize(Array(1, 1, length, length))
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!output.isEmpty && output.nElement() == input.nElement()) return output
    output.resize(input.size(2), input.size(2)).zero()
    attentionBiasLowerTriangle[T](input.size(2), output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isEmpty && gradInput.nElement() == input.nElement()) return gradInput
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

private[nn] class TransformerPrepareDecoder[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  @transient private var rangeBuffer : Tensor[T] = null
  @transient private var timeBuffer : Tensor[T] = null

  private def initRangeTensor(length: Int) = {
    if (rangeBuffer == null) {
      rangeBuffer = Tensor[T]().resize(Array(length, 2))
      val arr = rangeBuffer.select(2, 1).storage().array()
      for (i <- 0 to (length - 1)) {
        arr(i * 2) = ev.fromType(i)
        arr(i * 2 + 1) = ev.fromType(i)
      }
    }
  }

  /**
   * x: a Tensor with shape [batch, length, channels]
   * min_timescale: a float
   * max_timescale: a float
   * Returns: a Tensor the same shape as x.
   * @param input
   * @param min_timescale
   * @param max_timescale
   * @return
   */
  def addTimingSignal1D(input: Tensor[T],
     min_timescale : Float = 1.0f,
     max_timescale: Float = 1.0e4f): Tensor[T] = {
    if (timeBuffer != null) return timeBuffer
    // first dim is batch
    val length = input.size(2)
    val channels = input.size(3)
    // get_timing_signal_1d, return (1, length, channels)
    val num_timescales = channels / 2
    val log_timescale_increment = math.log(max_timescale / min_timescale) /
      math.max(num_timescales - 1, 1)
    // tf.range(num_timescales)
    val inv_timescales = new Array[Double](num_timescales)
    var i = 0
    while (i < inv_timescales.length) {
      inv_timescales(i) = min_timescale * math.exp(i * - log_timescale_increment)
      i += 1
    }
    rangeBuffer.select(2, 1).mul(ev.fromType[Double](inv_timescales(0)))
    rangeBuffer.select(2, 2).mul(ev.fromType[Double](inv_timescales(1)))

    val sinRes = rangeBuffer.clone().apply1(e =>
      ev.fromType(math.sin(ev.toType[Float](e))))
    val cosRes = rangeBuffer.clone().apply1(e =>
      ev.fromType(math.cos(ev.toType[Float](e))))

    if (timeBuffer == null) timeBuffer = Tensor[T](length, channels)
    timeBuffer.narrow(2, 1, sinRes.size(2)).copy(sinRes)
    timeBuffer.narrow(2, sinRes.size(2) + 1, cosRes.size(2)).copy(cosRes)
    timeBuffer
  }

  // input a Tensor with shape [batch, length, channels]
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    TransformerOperation.shiftRight3D(input, output)
    initRangeTensor(input.size(2))
    addTimingSignal1D(output)
    val batchSize = input.size(1)
    var i = 1
    while (i <= batchSize) {
      output.select(1, i).add(timeBuffer)
      i += 1
    }
    return output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (gradInput == null) gradInput = Tensor[T]()
    gradInput.resizeAs(gradOutput).zero()
    val size = gradOutput.size(2)
    var i = 1
    while (i < size) {
      gradInput.select(2, i).copy(gradOutput.select(2, i + 1))
      i += 1
    }
    gradInput
  }
}
