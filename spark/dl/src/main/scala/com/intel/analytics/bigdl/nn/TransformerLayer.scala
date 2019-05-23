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
   val problem: TransformerType = LanguageModel)
  (implicit ev: TensorNumeric[T]) extends BaseModule[T] {

  override def buildModel(): Module[T] = {
    problem match {
      case LanguageModel => buildLM()
      case Translation => buildTranslation()
    }
  }

  // input: int tensor with shape [batch_size, input_length].
  // target: int tensor with shape [batch_size, target_length].
  private def buildTranslation(): Module[T] = {
    val inputNode = Input()
    val targetNode = Input()
    val attentionBias = new PaddingMask().inputs(inputNode)

    val embedding = LookupTable[T](vocabSize, hiddenSize)
    val embeddingDupicate = LookupTable[T](vocabSize, hiddenSize)
    // parameter share
    val params1 = embedding.getParameters()
    val params2 = embeddingDupicate.getParameters()
    params1._1.set(params2._1)
    params1._2.set(params2._2)

    val constantValue = math.sqrt(hiddenSize)
    val embeddingInput = MulConstant(constantValue).inputs(embedding.inputs(inputNode))
    val embeddingOutput = MulConstant(constantValue).inputs(embeddingDupicate.inputs(targetNode))
    val encoderOutput = encode(embeddingInput, attentionBias)
    val outputNode = decode(embeddingOutput, encoderOutput, attentionBias)
    Graph(Array(inputNode, targetNode), outputNode)
  }

  private def buildLM(): Module[T] = {
    val inputNode = Input()
    val constantValue = math.sqrt(hiddenSize)
    val embeddingInput = MulConstant(constantValue).inputs(
      LookupTable[T](vocabSize, hiddenSize).inputs(inputNode))
    val outputNode = decode(embeddingInput)
    Graph(inputNode, outputNode)
  }

  private[nn] def encode(inputs: ModuleNode[T], attentionBias: ModuleNode[T]): ModuleNode[T] = {
    // Prepare inputs to the layer stack by adding positional encodings and
    // applying dropout.
    val position = new PositionEncode().inputs(inputs)
    val encoderInput = CAddTable().inputs(inputs, position)
    val decoderInputDrop = if (train) {
      val postDropOut = Dropout(1- postprocessDropout)
      postDropOut.inputs(encoderInput)
    } else encoderInput

    encodeStack(numHiddenlayers, decoderInputDrop, attentionBias)
  }

  private[nn] def decode(targets: ModuleNode[T],
                     encoderOutput: ModuleNode[T] = null,
                     attentionBias: ModuleNode[T] = null): ModuleNode[T] = {
    val decoderInput = new PositionEncodeWithShift().inputs(targets)
    val decoderSelfAttentionBias = new SelfAttentionMask().inputs(targets)

    val decoderInputDrop = if (train) {
      val postDropOut = Dropout(1- postprocessDropout)
      postDropOut.inputs(decoderInput)
    } else decoderInput

    decodeStack(numHiddenlayers, decoderInputDrop,
      decoderSelfAttentionBias, encoderOutput, attentionBias)
  }


  private[nn] def encodeStack(numLayers: Int,
                              encoderInput: ModuleNode[T],
                              attentionBias: ModuleNode[T]): ModuleNode[T] = {
    decodeStack(numLayers, encoderInput, attentionBias, preName = "encode")
  }

  private[nn] def decodeStack(numLayers: Int,
                              decoderInput: ModuleNode[T],
                              decoderSelfAttentionBias: ModuleNode[T],
                              encoderOutputs: ModuleNode[T] = null,
                              attentionBias: ModuleNode[T] = null,
                              preName: String = "decode"): ModuleNode[T] = {
    var input = decoderInput
    var i = 0
    while (i < numLayers) {
      val selfAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
      val selfAttentionModel = prePostProcessingSelfAttention(
        selfAttention, input, decoderSelfAttentionBias,
        s"${preName}_self_attention_${i}")
      input = selfAttentionModel

      if (encoderOutputs != null && attentionBias != null) {
        val encdecAttention = new Attention[T](hiddenSize, numHeads, attentionDropout)
        val encdecAttentionModel = prePostProcessingEncDecAttention(
          encdecAttention, input, encoderOutputs, attentionBias,
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

  private def prePostProcessingSelfAttention(layer: Module[T], decoderInput: ModuleNode[T],
    decoderSelfAttentionBias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
        .inputs(decoderInput)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
        .inputs(layer.setName(preName + "/self_attention")
        .inputs(norm, norm, decoderSelfAttentionBias))
    CAddTable().inputs(decoderInput, drop)
  }

  private def prePostProcessingEncDecAttention(
    layer: Module[T],
    decoderInput: ModuleNode[T],
    encoderOutput: ModuleNode[T],
    attentionBias: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoderInput)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/encdec_attention")
        .inputs(norm, encoderOutput, attentionBias))
    CAddTable().inputs(decoderInput, drop)
  }

  private def prePostProcessingFFN(layer: Module[T],
    decoderInput: ModuleNode[T], preName: String): ModuleNode[T] = {
    val norm = new LayerNormalization[T](hiddenSize).setName(preName + "/norm")
      .inputs(decoderInput)
    val drop = Dropout[T](1 - postprocessDropout).setName(preName + "/dropout")
      .inputs(layer.setName(preName + "/ffn").inputs(norm))
    CAddTable().inputs(decoderInput, drop)
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
     problem: TransformerType = LanguageModel)
   (implicit ev: TensorNumeric[T]): TransformerLayer[T] =
    new TransformerLayer(vocabSize, hiddenSize, numHeads,
      filterSize, numHiddenlayers,
      postprocessDropout, attentionDropout, reluDropout, problem)
}

/**
 * Return positional encoding.
 * Calculates the position encoding as a mix of sine and cosine functions with
 * geometrically increasing wavelengths.
 * Defined and formulized in Attention is All You Need, section 3.5.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
private[nn] class PositionEncode[T: ClassTag](implicit ev: TensorNumeric[T])
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

// Return postition encoding with input shift right
private[nn] class PositionEncodeWithShift[T: ClassTag](implicit ev: TensorNumeric[T])
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

/**
 * Calculate bias tensor from padding values in tensor.
 * Bias tensor that is added to the pre-softmax multi-headed attention logits,
 * which has shape [batch_size, num_heads, length, length]. The tensor is zero at
 * non-padding locations, and -1e9 (negative infinity) at padding locations.
 * @param ev$1
 * @param ev
 * @tparam T The numeric type in this module parameters
 */
private[nn] class PaddingMask[T: ClassTag](implicit ev: TensorNumeric[T])
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

// This mask is to hide both <pad> and future words. Used in decode
private[nn] class SelfAttentionMask[T: ClassTag](implicit ev: TensorNumeric[T])
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
        // reminder: here not 1
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