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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, Initializable}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.{SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.collection.mutable
import scala.reflect.ClassTag

/**
 *LookupTable for multi-values.
 * Also called embedding_lookup_sparse in TensorFlow.
 *
 * The input of LookupTableSparse should be a 2D SparseTensor or two 2D sparseTensors.
 * If the input is a SparseTensor, the values are positive integer ids,
 * values in each row of this SparseTensor will be turned into a dense vector.
 * If the input is two SparseTensors, the first tensor should be the integer ids, just
 * like the SparseTensor input. And the second tensor is the corresponding
 * weights of the integer ids.
 *
 * @param nIndex Indices of input row
 * @param nOutput the last dimension size of output
 * @param combiner A string specifying the reduce type.
 *                 Currently "mean", "sum", "sqrtn" is supported.
 * @param maxNorm If provided, each embedding is normalized to have l2 norm equal to
 *                maxNorm before combining.
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 */
class LookupTableSparse[T: ClassTag](
  val nIndex: Int, val nOutput: Int,
  val combiner: String = "sum",
  val maxNorm: Double = -1,
  var wRegularizer: Regularizer[T] = null)(
  implicit ev: TensorNumeric[T]) extends AbstractModule[Activity, Tensor[T], T] with Initializable {
  val weight = Tensor[T](nIndex, nOutput)
  val gradWeight = Tensor[T](nIndex, nOutput).zero()

  require(combiner == "mean" || combiner == "sum" || combiner == "sqrtn", "LookupTableSparse's" +
    s" combiner should be one of mean, sum or sqrtn, but got ${combiner}")

  protected val inputBuffer: Tensor[T] = Tensor()
  protected val inputWeightBuffer: Tensor[T] = Tensor()
  protected val frameBuffer: Tensor[T] = Tensor()
  protected val ids: Tensor[T] = Tensor()
  protected val indices: Tensor[Int] = Tensor[Int]()
  protected val batchScaleBuffer: Tensor[T] = Tensor[T]()
  protected var nonZeroCount: Array[Int] = _
  protected val normScale: mutable.HashMap[Int, T] = mutable.HashMap[Int, T]()

  {
    val wInit = RandomNormal(0, 1)
    setInitMethod(weightInitMethod = wInit)
  }

  override def reset(): Unit = {
    weightInitMethod.init(weight, VariableFormat.Default)
  }

  override def updateOutput(input: Activity): Tensor[T] = {
    val (inputTensor, weightTensor) = if (input.isTable) {
      (input.toTable[Tensor[T]](1), Some(input.toTable[Tensor[T]](2)))
    } else {
      (input.toTensor[T], None)
    }
    require(inputTensor.getTensorType == SparseType, "LookupTableSparse's input" +
      s"must be SparseTensor, but got ${inputTensor.getTensorType}")

    val batchSize = inputTensor.size(1)
    inputBuffer.set(inputTensor.storage(),
      inputTensor.storageOffset(),
      Array(inputTensor.nElement()))
    if (weightTensor.isDefined) {
      val weight = weightTensor.get
      inputWeightBuffer.set(weight.storage(),
        weight.storageOffset(),
        Array(weight.nElement()))
    }

    Tensor.unique(inputBuffer, ids, indices)

    if (maxNorm > 0) {
      normScale.clear()
      LookupTableSparse.norm2ScaleWithIndices[T](
          weight, ids, ev.fromType(maxNorm), normScale)
    }

    nonZeroCount = inputTensor.numNonZeroByRow()
    output.resize(batchSize, nOutput).zero()
    batchScaleBuffer.resize(batchSize)

    var i = 0 // index for all the ids in the input
    var b = 0
    while (b < batchSize) {
      val times = nonZeroCount(b)
      // compute a overall scale for this batch
      val batchScale = if (combiner == "sum") {
        // if combiner == sum, batchScale = 1
        ev.one
      } else {
        var count = times.toFloat
        if (weightTensor.isDefined) {
          count = 0
          var j = 0
          while (j < times) {
            if (combiner == "mean") {
              count += ev.toType[Float](inputWeightBuffer.valueAt(i + j + 1))
            } else {
              count += math.pow(ev.toType[Float](inputWeightBuffer.valueAt(i + j + 1)), 2).toFloat
            }
            j += 1
          }
        }
        if (combiner == "mean") {
          // if combiner == mean, batchScale = sum(inputWeightBuffer) / times
          ev.fromType(1f / count)
        } else {
          // if combiner == sqrtn, batchScale = sqrt(sum(inputWeightBuffer^2)) / times
          ev.fromType(1f / math.sqrt(count))
        }
      }
      // save this batchScale
      batchScaleBuffer.setValue(b + 1, batchScale)

      var j = 0
      while (j < times) {
        val index = ev.toType[Int](inputBuffer.valueAt(i + 1))
        // scale = normScale * batchScale * sp_weights
        val scale = ev.times(
          if (normScale != null && normScale.contains(index)) normScale(index) else ev.one,
          ev.times(batchScale,
            if (weightTensor.isDefined) inputWeightBuffer.valueAt(i + 1) else ev.one))
        // output += scale * weight(index)
        output.select(1, b + 1).add(scale, weight.select(1, index))
        i += 1
        j += 1
      }
      b += 1
    }

    output
  }

  override def updateGradInput(input: Activity, gradOutput: Tensor[T]): Activity = {
    // Input is not derivable
    gradInput
  }

  override def accGradParameters(input: Activity, gradOutput: Tensor[T]): Unit = {
    val batchSize = output.size(1)
    val three = ev.fromType(3)

    var b = 0
    var i = 0
    while (b < batchSize) {
      val times = nonZeroCount(b)
      var j = 0
      while (j < times) {
        val index = ev.toType[Int](inputBuffer.valueAt(i + 1))
        val gradWeightFrame = gradWeight.select(1, index)
        val gradOutputFrame = gradOutput.select(1, b + 1)
        // scale = normScale * batchScale * sp_weights
        val scale = ev.times(
          if (normScale != null) normScale.getOrElse(index, ev.one) else ev.one,
          ev.times(batchScaleBuffer.valueAt(b + 1),
            if (!inputWeightBuffer.isEmpty) inputWeightBuffer.valueAt(i + 1) else ev.one))
        // gradWeight += gradOutput * scale
        gradWeightFrame.add(scale, gradOutputFrame)

        // if norm2 clipping is invoked, need to compute the clipping's gradient.
        if (normScale != null && normScale.contains(index)) {
          val weightFrame = weight.select(1, index)
          // sum = sum(weightFrame * gradOutputFrame) * maxNorm * sp_weights * batchScale
          val sum = ev.times(frameBuffer.resizeAs(weightFrame).copy(weightFrame)
            .cmul(gradOutputFrame).sum,
            ev.times(ev.fromType(maxNorm), ev.divide(scale, normScale(index))))
          // gradWeight += - (normScale / maxNorm)^3 * sum * gradOutput
          gradWeightFrame.add(ev.times(sum, ev.negative(
            ev.pow(ev.divide(normScale(index), ev.fromType(maxNorm)), three))),
            weight.select(1, index))
        }
        i += 1
        j += 1
      }
      b += 1
    }

    if (null != wRegularizer) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
  }

  override def toString(): String = {
    val s = s"${getPrintName}" +
      s"(nIndex=$nIndex,nOutput=$nOutput,"
    if (maxNorm > 0) {
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
    inputWeightBuffer.set()
    frameBuffer.set()
    ids.set()
    indices.set()
    batchScaleBuffer.set()
    nonZeroCount = null
    normScale.clear()
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
        maxNorm == that.maxNorm
    case _ => false
  }

  override def hashCode(): Int = {
    def getHashCode(a: Any): Int = if (a == null) 0 else a.hashCode()
    val state = Seq(super.hashCode(), weight, gradWeight, nIndex, nOutput, maxNorm)
    state.map(getHashCode).foldLeft(0)((a, b) => 31 * a + b)
  }

}

object LookupTableSparse {
  def apply[T: ClassTag](
    nIndex: Int, nOutput: Int,
    combiner: String = "sum",
    maxNorm: Double = -1,
    wRegularizer: Regularizer[T] = null)(
    implicit ev: TensorNumeric[T]): LookupTableSparse[T] = {
    new LookupTableSparse(nIndex, nOutput, combiner.toLowerCase,
      maxNorm, wRegularizer)
  }

  /**
   * Compute the l2 norm clipping scale of the indices-th frame in tensor's first dimension.
   * @return a HashMap contains l2 norm clipping scale
   */
  protected def norm2ScaleWithIndices[T: ClassTag](
      tensor: Tensor[T],
      indices: Tensor[T],
      maxNorm: T,
      scaleBuffer: mutable.HashMap[Int, T])(
      implicit ev: TensorNumeric[T]): mutable.HashMap[Int, T] = {
    val indicesArray = indices.storage.array()
    var i = indices.storageOffset() - 1
    while (i < indices.nElement() + indices.storageOffset() - 1) {
      val index = ev.toType[Int](indicesArray(i))
      val norm = tensor(index).norm(2)
      if (ev.isGreater(norm, maxNorm)) scaleBuffer(index) = ev.divide(maxNorm, norm)
      i += 1
    }

    scaleBuffer
  }

}
