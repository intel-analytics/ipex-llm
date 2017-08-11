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

package com.intel.analytics.bigdl.nn.quantization

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.nn.{ErrorInfo, Module}
import com.intel.analytics.bigdl.quantization.Quantization
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{FloatType, QuantizeTensor, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import java.nio.ByteBuffer
import java.nio.file.{Files, Paths}
import scala.reflect.ClassTag

class Linear[T: ClassTag](
  inputSize: Int,
  outputSize: Int,
  withBias: Boolean = true
)(implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  val data: QuantizeTensor[T] = QuantizeTensor[T]()
  val weight: QuantizeTensor[T] = QuantizeTensor[T]()
  val bias: Tensor[T] = Tensor[T](outputSize)

  val weightSum = new Array[T](outputSize)
  val min = new Array[T](outputSize)
  val max = new Array[T](outputSize)

  val FAULT_TOLERANCE = 0.5f
  val WEIGHT_THRESHOLD = 64.0f
  val THRESHOLD = 127.0f

  @transient
  var _init = false

  private def setWeightSum(weight: Tensor[T], weightSum: Array[T]): Unit = {
    for (i <- 1 to outputSize) {
      val singleRow = weight.select(1, i)
      weightSum(i - 1) = singleRow.sum()
    }
  }

  def initWeightAndBias(weightFP32: Tensor[T], biasFP32: Tensor[T]): this.type = {
    if (biasFP32 != null) {
      bias.copy(biasFP32)
    } else {
      bias.fill(ev.fromType(0)) // TODO bias may be null, at that time, we should not initialize it
    }

    val weightFP32Tmp = weightFP32.view(Array(outputSize, inputSize))
    setWeightSum(weightFP32Tmp, weightSum)

    for (i <- 1 to outputSize) {
      val singleRow = weightFP32Tmp.select(1, i)
      min(i - 1) = singleRow.min()
      max(i - 1) = singleRow.max()
    }

    val bufferOffset = 0
    val buffer = ByteBuffer.allocate(weightFP32.nElement())
    val weightFP32Tensor = weightFP32.asInstanceOf[Tensor[Float]]
    Quantize.quantize(weightFP32Tensor, buffer, bufferOffset)

    weight.setStorage(buffer)

    init()

    this
  }

  def init(): this.type = {
    val byteArrayOfWeight = weight.getStorage.get
    weight.setStorageInJni(
      Quantization.FixFCKernelDescInit(outputSize, inputSize))

    ev.getType() match {
      case FloatType =>
        val minArray = min.asInstanceOf[Array[Float]]
        val maxArray = max.asInstanceOf[Array[Float]]

        Quantization.FixFCKernelLoadFromModel(weight.getStorageInJni, byteArrayOfWeight,
          minArray, maxArray, outputSize, inputSize, WEIGHT_THRESHOLD, Quantization.NCHW)
      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    _init = true

    this
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.dim() == 1 || input.dim() == 2,
      "Linear: " + ErrorInfo.constrainInputAsVectorOrBatch)

    if (!_init && weight.getStorage.isDefined) {
      init()
    }

    val batchSize = if (input.dim() == 1) {
      output.resize(Array(outputSize)) // TODO
      1
    } else {
      output.resize(Array(input.size(1), outputSize))
      input.size(1)
    }

    if (!data.isInitialized) {
      data.setStorageInJni(Quantization.FixFCDataDescInit(batchSize, inputSize))
    }

    ev.getType() match {
      case FloatType => // TODO
        val src = input.storage().array().asInstanceOf[Array[Float]]
        val offset = input.storageOffset() - 1

        Quantization.FixFCDataInit(data.getStorageInJni, src, offset, batchSize, inputSize,
          THRESHOLD, Quantization.NCHW)

        val outputArray = output.storage().array().asInstanceOf[Array[Float]]
        val outputOffset = output.storageOffset() - 1
        val weightSumArray = weightSum.asInstanceOf[Array[Float]]
        val weightSumOffset = 0
        val biasArray = bias.storage().array().asInstanceOf[Array[Float]]
        val biasOffset = bias.storageOffset() - 1

        Quantization.InternalMixPrecisionConvolutionGEMM(
          Quantization.NCHW, weight.getStorageInJni, data.getStorageInJni, outputArray, outputOffset,
          weightSumArray, weightSumOffset, biasArray, biasOffset,
          batchSize, outputSize, 1, 1,
          FAULT_TOLERANCE)

      case _ => throw new UnsupportedOperationException(s"Only support Float for quantized model")
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    throw new UnsupportedOperationException(s"Doesn't updateGradInput for quantized model")
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(null, null), Array(null, null))
  }

  override def getParametersTable(): Table = {
    T(getName() -> T("weight" -> weight, "bias" -> bias))
  }

  override def equals(obj: Any): Boolean = {
    if (!super.equals(obj)) {
      return false
    }

    if (!obj.isInstanceOf[Linear[T]]) {
      return false
    }
    val other = obj.asInstanceOf[Linear[T]]
    if (this.eq(other)) {
      return true
    }

    weight == other.weight &&
      bias == other.bias
  }

  override def hashCode(): Int = {
    val seed = 37
    var hash = super.hashCode()
    hash = hash * seed + weight.hashCode()
    hash = hash * seed + bias.hashCode()

    hash
  }

  override def clearState() : this.type = {
    super.clearState()
    this
  }

  override def toString(): String = {
    s"fixpoint.${getPrintName()}($inputSize -> $outputSize)"
  }

  def release(): Unit = {
    weight.release()
    data.release()
  }
}

object TestLinear {
  case class TestCase(batchSize: Int, inputSize: Int, outputSize: Int)
  def main(args: Array[String]): Unit = {
    import com.intel.analytics.bigdl.nn.{Linear => NNLinear}
    val test = TestCase(2, 4, 3)

    val weight = Tensor[Float](test.outputSize, test.inputSize)

    for (i <- 0 until weight.nElement()) {
      weight.storage().array()(i) = i % 32
    }
    val bias = Tensor[Float](test.outputSize).fill(0f)

    val nnLinear = new NNLinear[Float](test.inputSize, test.outputSize, initBias = bias)

    val input = Tensor[Float]().resize(Array(test.batchSize, test.inputSize))
    for (i <- 0 until input.nElement()) {
      input.storage().array()(i) = i % 32
    }

    val quantizedLinear = new Linear[Float](test.inputSize, test.outputSize)

    nnLinear.updateOutput(input)
    quantizedLinear.initWeightAndBias(nnLinear.weight, nnLinear.bias)
    quantizedLinear.updateOutput(input)

    Files.deleteIfExists(Paths.get("/tmp/quantizedLinear"))
    quantizedLinear.save("/tmp/quantizedLinear")

    val tmp = Module.load("/tmp/quantizedLinear").asInstanceOf[Linear[Float]]
    println(tmp)
    tmp.updateOutput(input)

    Files.deleteIfExists(Paths.get("/tmp/nnLinear"))
    nnLinear.save("/tmp/nnLinear")

    val newInput = Tensor[Float]().resize(Array(test.batchSize, test.inputSize))
    for (i <- 0 until input.nElement()) {
      newInput.storage().array()(i) = i % 32
    }

    quantizedLinear.updateOutput(newInput)
    nnLinear.updateOutput(newInput)
    tmp.updateOutput(newInput)

    println(tmp.output.nElement())
    println(quantizedLinear.output.nElement())
    println(nnLinear.output.nElement())

    require(tmp.output.nElement() == quantizedLinear.output.nElement(),
      s"elements number should be the same")

    for (i <- 0 until tmp.output.nElement()) {
      val ori = quantizedLinear.output.storage().array()(i)
      val ser = tmp.output.storage().array()(i)

      require(Math.abs(ori - ser) < 0.1, s"values should be the same.")
    }

    quantizedLinear.release()

  }
}

object Linear {
  def apply[@specialized(Float, Double) T: ClassTag](
    inputSize: Int,
    outputSize: Int,
    withBias: Boolean = true
  )(implicit ev: TensorNumeric[T]) : Linear[T] = {
    new Linear[T](inputSize, outputSize, withBias)
  }
}
