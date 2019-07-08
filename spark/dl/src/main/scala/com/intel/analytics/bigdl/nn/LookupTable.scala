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

import breeze.numerics.{abs, pow}
import com.intel.analytics.bigdl.nn.abstractnn.{Initializable, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.{Shape, T, Table}

import scala.reflect.ClassTag

/**
 * This layer is a particular case of a convolution, where the width of the convolution would be 1.
 * Input should be a 1D or 2D tensor filled with indices. Indices are corresponding to the position
 * in weight. For each index element of input, it outputs the selected index part of weight.
 * Elements of input should be in range of (1, nIndex)
 * This layer is often used in word embedding.
 * @param nIndex Indices of input row
 * @param nOutput the last dimension size of output
 * @param paddingValue padding value, default 0
 * @param maxNorm max norm, defalt Double.MaxValue
 * @param normType norm regularization number, default 2
 * @param shouldScaleGradByFreq
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param maskZero: if maskZero is set to true, the input whose value equals `paddingValue`
 *                the output will be masked to zero vector.
 */
@SerialVersionUID( - 4832171200145114633L)
class LookupTable[T: ClassTag]
(val nIndex: Int, val nOutput: Int, val paddingValue: Double = 0,
  val maxNorm: Double = Double.MaxValue,
  val normType: Double = 2.0,
  shouldScaleGradByFreq: Boolean = false,
  var wRegularizer: Regularizer[T] = null,
  val maskZero: Boolean = false
)
(implicit ev: TensorNumeric[T]) extends TensorModule[T] with Initializable {

  var weight = Tensor[T](nIndex, nOutput)
  var gradWeight = Tensor[T](nIndex, nOutput).zero()

  private var inputBuffer = Tensor[T]()
  private var normBuffer = Tensor[T]()
  private val countBuffer = Tensor[T]()

  {
    val wInit = RandomNormal(0, 1)
    setInitMethod(weightInitMethod = wInit)
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.Default)
  }

  private def renorm(input : Tensor[T]): Unit = {
    if (Double.MaxValue == maxNorm) {
      return
    }
    normBuffer.resize(input.size()).copy(input)
    if (normBuffer.dim() == 2) {
      normBuffer = normBuffer.view(normBuffer.nElement())
    }
    require(weight.isContiguous(), "LookupTable: weight must be contiguous")
    require(normBuffer.isContiguous(), "LookupTable: input must be contiguous")
    require(normBuffer.nDimension() == 1, "LookupTable: idx must be a vector")
    require(normType > 0, "LookupTable: non-positive-norm not supported")

    val rowIdx = normBuffer.storage().array()
    val rowOffset = normBuffer.storageOffset() - 1
    var numEle = normBuffer.nElement()
    val stride = weight.stride(1)

    val gw = weight.storage().array()
    val gw_offset = weight.storageOffset() - 1

    var i = 0
    while (i < numEle) {
      require(ev.isGreater(ev.fromType(weight.size(1) + 1), rowIdx(i + rowOffset)),
        s"LookupTable: elements of input should be little than or equal to $nIndex + 1")
      require(ev.isGreaterEq(rowIdx(i + rowOffset), ev.one),
        "LookupTable: elements of input should be greater than or equal to 1")
      i += 1
    }

    implicit val ord = Ordering.fromLessThan[T]((e1, e2) => (ev.isGreater(e1, e2)))
    scala.util.Sorting.quickSort(rowIdx)

    var ptr = 0
    i = 0
    while (i < numEle) {
      if (i == 0 || rowIdx(i + rowOffset) != rowIdx(i - 1 + rowOffset)) {
        rowIdx(ptr + rowOffset) = rowIdx(i + rowOffset)
        ptr += 1
      }
      i += 1
    }
    numEle = ptr

    i = 0
    while (i < numEle) {
      val k = ev.toType[Int](rowIdx(i + rowOffset)) - 1
      renormRow(gw, k * stride + gw_offset, stride, maxNorm, normType)
      i += 1
    }
  }

  private def renormRow(row_data: Array[T], offset: Int, stride: Int,
                        maxNorm: Double, normType: Double): Unit = {
    var norm = 0.0
    var j = 0
    while (j < stride) {
      if (normType == 1) {
        norm += ev.toType[Double](ev.abs(row_data(j + offset)))
      } else if (normType == 2) {
        norm += ev.toType[Double](ev.times(row_data(j + offset), row_data(j + offset)))
      } else {
        norm += math.pow(abs(ev.toType[Double](row_data(j + offset))), normType)
      }
      j += 1
    }
    norm = pow(norm, 1.0 / normType)

    // Keep the norm of weight smaller than maxNorm
    if (norm > maxNorm) {
      val new_norm = maxNorm / (norm + 1e-7)
      j = 0
      while (j < stride) {
        row_data(j + offset) = ev.times(row_data(j + offset), ev.fromType(new_norm))
        j += 1
      }
    }
  }

  private def resetCount(count: Tensor[T], input: Tensor[T]): Unit = {
    var i = 1
    val numEle = input.nElement()

    while (i <= numEle) {
      val k = ev.toType[Int](input.valueAt(i))
      count.update(k, ev.zero)
      i += 1
    }

    i = 1
    while (i <= numEle) {
      val k = ev.toType[Int](input.valueAt(i))
      count.update(k, ev.plus(count.valueAt(k), ev.one))
      i += 1
    }
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    if (maskZero && paddingValue != 0) {
      weight.select(1, paddingValue.toInt).zero()
    }
    require(input.dim() == 1 || input.dim() == 2,
      s"LookupTable: ${ErrorInfo.constrainInputAsVectorOrBatch}, input dim [${input.dim()}]"  )
    renorm(input)
    inputBuffer = input.contiguous()
    try {
      if (inputBuffer.dim() == 1) {
        output.index(1, inputBuffer, weight)
      } else if (inputBuffer.dim() == 2) {
        output.index(1, inputBuffer.view(inputBuffer.nElement()), weight)
        output = output.view(inputBuffer.size(1), inputBuffer.size(2), weight.size(2))
      }
    } catch {
      case e: IllegalArgumentException =>
        throw new IllegalArgumentException(
          s"LookupTable updateOutput get exception:${e.getMessage}\n" +
          s"please ensure elements of your input will not exceed ${nIndex}")
      case e: Exception =>
        throw e
    }

    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (!gradInput.isSameSizeAs(input)) {
      gradInput.resizeAs(input).zero()
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    inputBuffer = input.contiguous()
    require(gradWeight.isContiguous(), "LookupTable: gradWeight must be contiguous")
    require(inputBuffer.dim() == 1 || inputBuffer.dim() == 2,
      s"LookupTable: input must be a vector or matrix, input dim ${inputBuffer.dim()}" )

    if (inputBuffer.dim() == 2) {
      inputBuffer.view(inputBuffer.nElement())
    }
    val _gradOutput = gradOutput.contiguous()
    var count_data : Array[T] = null
    if (shouldScaleGradByFreq) {
      countBuffer.resize(gradWeight.size(1))
      resetCount(countBuffer, inputBuffer)
      count_data = countBuffer.storage().array()
    }

    val input_data = inputBuffer.storage().array()
    val input_offset = inputBuffer.storageOffset() - 1
    val numEle = inputBuffer.nElement()

    var i = 0
    while (i < numEle) {
      require(ev.isGreater(ev.fromType(gradWeight.size(1) + 1), input_data(i + input_offset)),
        s"LookupTable: elements of input should be little than or equal to $nIndex + 1")
      require(ev.isGreaterEq(input_data(i + input_offset), ev.one),
        "LookupTable: elements of input should be greater than or equal to 1")
      i += 1
    }
    if (scaleW != 0) {
      val gw = gradWeight.storage().array()
      val go = _gradOutput.storage().array()
      val stride = gradWeight.stride(1)

      i = 0
      while (i < numEle) {
        if (input_data(i + input_offset) != paddingValue) {
          val k = ev.toType[Int](input_data(i + input_offset)) - 1
          val scale_ = if (null != count_data) scaleW /
            ev.toType[Double](count_data(k)) else scaleW
          ev.axpy(stride, ev.fromType(scale_), go, i * stride + _gradOutput.storageOffset() - 1, 1,
            gw, k * stride + gradWeight.storageOffset() - 1, 1)
        }
        i += 1
      }

      if (null != wRegularizer) {
        wRegularizer.accRegularization(weight, gradWeight, scaleW)
      }
    }
  }

  override def toString(): String = {
    val s = s"${getPrintName}" +
      s"(nIndex=$nIndex,nOutput=$nOutput,paddingValue=$paddingValue,normType=$normType"
    if (maxNorm == Double.MaxValue) {
      s + ")"
    } else {
      s + s" ,maxNorm=$maxNorm)"
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight), Array(this.gradWeight))
  }

  override def clearState() : this.type = {
    super.clearState()
    inputBuffer.set()
    countBuffer.set()
    normBuffer.set()
    this
  }

  override def canEqual(other: Any): Boolean = other.isInstanceOf[LookupTable[T]]

  override def equals(other: Any): Boolean = other match {
    case that: LookupTable[T] =>
      super.equals(that) &&
        (that canEqual this) &&
        weight == that.weight &&
        gradWeight == that.gradWeight &&
        nIndex == that.nIndex &&
        nOutput == that.nOutput &&
        paddingValue == that.paddingValue &&
        maxNorm == that.maxNorm &&
        normType == that.normType
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), weight, gradWeight, nIndex, nOutput,
      paddingValue, maxNorm, normType)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }
  override def computeOutputShape(inputShape: Shape): Shape = {
    val _inputSize = inputShape.toSingle().toArray
    if (_inputSize.length == 2) {
      Shape(Array(_inputSize(0), _inputSize(1), nOutput))
    } else Shape(Array(_inputSize(0), nOutput))
  }
}

object LookupTable {
  def apply[@specialized(Float, Double)T: ClassTag](
    nIndex: Int, nOutput: Int,
    paddingValue: Double = 0, maxNorm: Double = Double.MaxValue,
    normType: Double = 2.0, shouldScaleGradByFreq: Boolean = false,
    wRegularizer: Regularizer[T] = null,
    maskZero: Boolean = false
  )
   (implicit ev: TensorNumeric[T]): LookupTable[T] =
    new LookupTable[T](nIndex, nOutput, paddingValue,
      maxNorm, normType, shouldScaleGradByFreq, wRegularizer, maskZero)
}


