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

import breeze.linalg.*
import com.intel.analytics.bigdl.Module
import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.T

import scala.collection.mutable.ArrayBuffer
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

  override def buildModel(): Module[T] = {
    problem match {
      case LanguageModel => buildLM()
      case Translation => buildTranslation()
    }
  }

  // inputs: int tensor with shape [batch_size, input_length].
  // targets: None or int tensor with shape [batch_size, target_length].
  private def buildTranslation(): Module[T] = {
    val inNode = Input()
    val tarNode = Input()
    val attention_bias = new AttentionBiasConstant().inputs(inNode)

    val embedding = LookupTable[T](vocabSize, hiddenSize)
    val embeddingDupicate = LookupTable[T](vocabSize, hiddenSize)
    // parameter share
    val params1 = embedding.getParameters()
    val params2 = embeddingDupicate.getParameters()
    params1._1.set(params2._1)
    params1._2.set(params2._2)

    val constantValue = math.sqrt(hiddenSize)
    val embeddingInput = MulConstant(constantValue).inputs(embedding.inputs(inNode))
    val embeddingOutput = MulConstant(constantValue).inputs(embeddingDupicate.inputs(tarNode))
    val encoder_outputs = encode(embeddingInput, attention_bias)
    val outNode = decode(embeddingOutput, encoder_outputs, attention_bias)
    Graph(Array(inNode, tarNode), outNode)
  }

  private def buildLM(): Module[T] = {
    val inNode = Input()
    val constantValue = math.sqrt(hiddenSize)
    val embeddingInput = MulConstant(constantValue).inputs(
      LookupTable[T](vocabSize, hiddenSize).inputs(inNode))
    val outNode = decode(embeddingInput)
    Graph(inNode, outNode)
  }

  private[nn] def encode(inputs: ModuleNode[T], attention_bias: ModuleNode[T]): ModuleNode[T] = {
    // Prepare inputs to the layer stack by adding positional encodings and
    // applying dropout.
    val input2 = new EncodePositionConstant().inputs(inputs)
    val encoder_inputs = CAddTable().inputs(inputs, input2)
    val decoder_input_drop = if (train) {
      val postDropOut = Dropout(1- postprocessDropout)
      postDropOut.inputs(encoder_inputs)
    } else encoder_inputs

    encodeStack(numHiddenlayers, decoder_input_drop, attention_bias)
  }

  private[nn] def decode(targets: ModuleNode[T],
                     encoder_outputs: ModuleNode[T] = null,
                     attention_bias: ModuleNode[T] = null): ModuleNode[T] = {
    val decoder_input = new TransformerPrepareDecoder().inputs(targets)
    val decoder_self_attention_bias = new SelfAttentionBiasConstant().inputs(targets)

    val decoder_input_drop = if (train) {
      val postDropOut = Dropout(1- postprocessDropout)
      postDropOut.inputs(decoder_input)
    } else decoder_input

    decodeStack(numHiddenlayers, decoder_input_drop,
      decoder_self_attention_bias, encoder_outputs, attention_bias)
  }


  private[nn] def encodeStack(num_layers: Int,
                              encoder_input: ModuleNode[T],
                              attention_bias: ModuleNode[T]): ModuleNode[T] = {
    decodeStack(num_layers, encoder_input, attention_bias, preName = "encode")
  }

  private[nn] def decodeStack(num_layers: Int,
                              decoder_input: ModuleNode[T],
                              decoder_self_attention_bias: ModuleNode[T],
                              encoder_outputs: ModuleNode[T] = null,
                              attention_bias: ModuleNode[T] = null,
                              preName: String = "decode"): ModuleNode[T] = {
    var input = decoder_input
    var i = 0
    while (i < num_layers) {
      val selfAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
      val selfAttentionModel = prePostProcessingSelfAttention(
        selfAttention, input, decoder_self_attention_bias,
        s"${preName}_self_attention_${i}")
      input = selfAttentionModel

      if (encoder_outputs != null && attention_bias != null) {
        val encdecAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
        val encdecAttentionModel = prePostProcessingEncDecAttention(
          encdecAttention, input, encoder_outputs, attention_bias,
          s"${preName}_encdec_attention_${i}")
        input = encdecAttentionModel
      }

      val ffn = new FeedForwardNetwork[T](hiddenSize, filterSize, reluDropout)
      val ffnModel = prePostProcessingFFN(ffn, input, s"${preName}_ffn_${i}")
      input = ffnModel

      i += 1
    }
    val norm = new LayerNormalization[T](hiddenSize).inputs(input)
    norm
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

  private def prePostProcessingEncDecAttention(
    layer: Module[T],
    decoder_input: ModuleNode[T],
    encoder_outputs: ModuleNode[T],
    attention_bias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoder_input)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/encdec_attention")
        .inputs(norm, encoder_outputs, attention_bias))
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
     reluDropout: Float,
      problem: ProblemType = LanguageModel)
   (implicit ev: TensorNumeric[T]): TransformerLayer[T] =
    new TransformerLayer(vocabSize, hiddenSize, numHeads,
      filterSize, numHiddenlayers,
      postprocessDropout, attentionDropout, reluDropout, problem)
}

private[nn] class EncodePositionConstant[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  @transient private var rangeBuffer : Tensor[T] = null

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (!output.isEmpty && output.nElement() == input.nElement()) return output
    val length = input.size(2)
    val channel = input.size(3)

    if (rangeBuffer == null) {
      rangeBuffer = Tensor[T]()
      TransformerOperation.initRangeTensor(length, rangeBuffer)
    }

    output.resize(length, channel)
    TransformerOperation.addTimingSignal1D(length, channel,
      rangeBuffer = rangeBuffer, timeBuffer = output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isEmpty && gradInput.nElement() == input.nElement()) return gradInput
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

private[nn] class AttentionBiasConstant[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    output.resizeAs(input).copy(input)
    output = TransformerOperation.getPaddingBias(output)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    gradInput.resizeAs(input).zero()
    gradInput
  }
}

private[nn] class TransformerPrepareDecoder[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  @transient private var rangeBuffer : Tensor[T] = null
  @transient private var timeBuffer : Tensor[T] = null

  // input a Tensor with shape [batch, length, channels]
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    TransformerOperation.shiftRight3D(input, output)
    val length = output.size(2)
    val channel = output.size(3)

    if (rangeBuffer == null) {
      rangeBuffer = Tensor[T]()
      TransformerOperation.initRangeTensor(length, rangeBuffer)
    }
    if (timeBuffer == null) {
      timeBuffer = Tensor[T]().resize(length, channel)
      TransformerOperation.addTimingSignal1D(length, channel,
        rangeBuffer = rangeBuffer, timeBuffer = timeBuffer)
    }
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

private[nn] class SelfAttentionBiasConstant[T: ClassTag](implicit ev: TensorNumeric[T])
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