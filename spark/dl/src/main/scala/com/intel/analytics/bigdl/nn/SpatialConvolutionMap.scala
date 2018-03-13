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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{DenseTensorConv, Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}

import scala.reflect.ClassTag

/**
 * This class is a generalization of SpatialConvolution.
 * It uses a generic connection table between input and output features.
 * The SpatialConvolution is equivalent to using a full connection table.
 *
 * @param wRegularizer: instance of [[Regularizer]]
 *                    (eg. L1 or L2 regularization), applied to the input weights matrices.
 * @param bRegularizer: instance of [[Regularizer]]
 *                    applied to the bias.
 */
@SerialVersionUID(5288662921102331388L)
class SpatialConvolutionMap[T: ClassTag](
  val connTable: Tensor[T],
  val kW: Int, // The kernel width of the convolution
  val kH: Int, // The kernel height of the convolution
  val dW: Int = 1, // The step of the convolution in the width dimension.
  val dH: Int = 1, // The step of the convolution in the height dimension
  val padW: Int = 0, // The additional zeros added per width to the input planes.
  val padH: Int = 0, // The additional zeros added per height to the input planes.
  var wRegularizer: Regularizer[T] = null,
  var bRegularizer: Regularizer[T] = null
)(implicit ev: TensorNumeric[T]) extends TensorModule[T]  {
  val nInputPlane = ev.toType[Int](connTable.select(2, 1).max())
  val nOutputPlane = ev.toType[Int](connTable.select(2, 2).max())
  val weight: Tensor[T] = Tensor[T](connTable.size(1), kH, kW)
  val bias: Tensor[T] = Tensor[T](nOutputPlane)
  val gradWeight: Tensor[T] = Tensor[T](connTable.size(1), kH, kW)
  val gradBias: Tensor[T] = Tensor[T](nOutputPlane)

  reset()

  // todo write a new InitializationMethod to wrap the following procedure
  override def reset(): Unit = {
    val ninp = Tensor[T](this.nOutputPlane).zero()
    var i = 1
    while (i <= connTable.size(1)) {
      ninp(Array(ev.toType[Int](connTable(Array(i, 2))))) = ev.plus(ninp(Array(ev.toType[Int](
        connTable(Array(i, 2))))), ev.fromType[Int](1))
      i = i + 1
    }

    var k = 1
    var stdv = ev.fromType[Int](0)
    while (k <= connTable.size(1)) {
      stdv = ev.divide(ev.fromType[Int](1), ev.sqrt(ev.times(ev.fromType[Int](kW * kH), ninp(
        Array(ev.toType[Int](connTable(Array(k, 2))))))))
      weight.select(1, k).apply1(_ => weight.uniform(ev.negative(stdv), stdv))
      k = k + 1
    }

    k = 1
    while (k <= bias.size(1)) {
      stdv = ev.divide(ev.fromType[Int](1), ev.sqrt(ev.times(ev.fromType[Int](kW * kH), ninp(
        Array(k)))))
      bias(k) = bias.uniform(ev.negative(stdv), stdv)
      k = k + 1
    }
    zeroGradParameters()
  }

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.nDimension() == 3 || input.nDimension() == 4,
      "3D or 4D(batch mode) tensor expected" +
        s"input dimension ${input.nDimension()}")
    val dimw = if (input.nDimension() == 4) 4 else 3
    val dimh = if (input.nDimension() == 4) 3 else 2
    val dimc = if (input.nDimension() == 4) 2 else 1
    val nbatch = if (input.nDimension() == 4) input.size(1) else 1
    require(input.size(dimc) >= nInputPlane, "invalid number of input planes" +
      s"input number ${input.size(dimc)}")
    require(input.size(dimw) >= kW && input.size(dimh) >= kH,
      "input smaller than kernel size" +
        s"input size (${input.size(dimw)},${input.size(dimh)}) " +
        s"kernel size (${kW},${kH})")

    val inputW = input.size(dimw)
    val inputH = input.size(dimh)
    val outputW = (inputW - kW) / dW + 1
    val outputH = (inputH - kH) / dH + 1

    // force batch
    if (input.nDimension() == 3) {
      output.resize(Array(1, nOutputPlane, outputH, outputW))
    } else {
      output.resize(Array(input.size(1), nOutputPlane, outputH, outputW))
    }

    val connTableIndex = new Array[Int](2)
    val outputIndex = new Array[Int](4)
    val biasIndex = new Array[Int](1)
    var m = 0
    while (m < nbatch) {
      var p = 0
      outputIndex(0) = m + 1
      while (p < nOutputPlane) {
        outputIndex(1) = p + 1
        biasIndex(0) = p + 1
        val z = bias(biasIndex)
        var j = 0
        while (j < outputH) {
          outputIndex(2) = j + 1
          var k = 0
          while (k < outputW) {
            outputIndex(3) = k + 1
            output(outputIndex) = z
            k += 1
          }
          j += 1
        }
        p += 1
      }

      val nWeight = connTable.size(1)
      var k = 1
      while (k <= nWeight) {
        connTableIndex(0) = k
        connTableIndex(1) = 2
        val o = ev.toType[Int](connTable(connTableIndex))
        connTableIndex(1) = 1
        val i = ev.toType[Int](connTable(connTableIndex))
        DenseTensorConv.validXCorr2Dptr[T](output.storage(),
          output.storageOffset() - 1 + (o - 1 + m * nOutputPlane) * outputH * outputW,
          ev.fromType[Int](1), input.storage(),
          input.storageOffset() - 1 + (i - 1 + m * nInputPlane) * inputW * inputH, inputH,
          inputW, weight.storage(), weight.storageOffset() - 1 + (k - 1) * kW * kH, kH, kW, dH, dW)
        k += 1
      }
      m += 1
    }

   if (input.nDimension() == 3) {
      output.squeeze(1)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dimw = if (input.nDimension() == 4) 4 else 3
    val dimh = if (input.nDimension() == 4) 3 else 2
    val nbatch = if (input.nDimension() == 4) input.size(1) else 1

    val inputW = input.size(dimw)
    val inputH = input.size(dimh)
    val weightH = weight.size(2)
    val weightW = weight.size(3)
    val outputH = gradOutput.size(dimh)
    val outputW = gradOutput.size(dimw)

    gradInput.resizeAs(input)
    gradInput.zero()

    val connTableIndex = new Array[Int](2)
    var m = 0
    while (m < nbatch) {
      val nkernel = connTable.size(1)
      var k = 1
      while (k <= nkernel) {
        connTableIndex(0) = k
        connTableIndex(1) = 2
        val o = ev.toType[Int](connTable(connTableIndex))
        connTableIndex(1) = 1
        val i = ev.toType[Int](connTable(connTableIndex))
        //        println(s"o:${o} i:${i}")
        DenseTensorConv.fullConv2Dptr(gradInput.storage().asInstanceOf[Storage[Double]],
          gradInput.storageOffset() - 1 + (i - 1 + m * nInputPlane) * inputW * inputH,
          1.0, gradOutput.storage().asInstanceOf[Storage[Double]], gradOutput.storageOffset() - 1
            + (o - 1 + m * nOutputPlane) * outputH * outputW,
          outputH, outputW, weight.storage().asInstanceOf[Storage[Double]], weight.storageOffset()
            - 1 + (k - 1) * weightW * weightH, weightH, weightW, dH, dW)
        //        print(gradInput)
        //        println()
        k += 1
      }
      m += 1
    }
    gradInput
  }

  override def accGradParameters(input: Tensor[T], gradOutput: Tensor[T]): Unit = {
    val dimw = if (input.dim() == 4) 4 else 3
    val dimh = if (input.dim() == 4) 3 else 2
    val nbatch = if (input.dim() == 4) input.size(1) else 1

    val inputW = input.size(dimw)
    val inputH = input.size(dimh)
    val outputW = gradOutput.size(dimw)
    val outputH = gradOutput.size(dimh)
    val weightH = weight.size(2)
    val weightW = weight.size(3)

    // force batch
    val forceBatch = if (gradOutput.nDimension() == 3) {
      gradOutput.addSingletonDimension()
      true
    } else {
      false
    }

    var m = 1
    if (scaleB != 0) {
      val gradOutputIndex = new Array[Int](4)
      val gradBiasIndex = new Array[Int](1)
      while (m <= nbatch) {
        gradOutputIndex(0) = m
        var k = 1
        while (k <= nOutputPlane) {
          gradOutputIndex(1) = k
          gradBiasIndex(0) = k
          var l = 1
          while (l <= outputH) {
            gradOutputIndex(2) = l
            var n = 1
            while (n <= outputW) {
              gradOutputIndex(3) = n
              gradBias(gradBiasIndex) = ev.plus(gradBias(gradBiasIndex),
                ev.times(ev.fromType[Double](scaleB), gradOutput(gradOutputIndex)))
              n += 1
            }
            l += 1
          }
          k += 1
        }
        m += 1
      }
    }

    m = 0
    if (scaleW != 0) {
      val nkernel = connTable.size(1)
      val connTableIndex = new Array[Int](2)
      while (m < nbatch) {
        var k = 1
        while (k <= nkernel) {
          connTableIndex(0) = k
          connTableIndex(1) = 2
          val o = ev.toType[Int](connTable(connTableIndex))
          connTableIndex(1) = 1
          val i = ev.toType[Int](connTable(connTableIndex))

          DenseTensorConv.validXCorr2DRevptr(gradWeight.storage(),
            gradWeight.storageOffset() - 1 + (k - 1) * weightH * weightW,
            ev.fromType[Double](scaleW), input.storage(),
            input.storageOffset() - 1 + (i - 1 + m * nInputPlane) * inputW * inputH,
            inputH, inputW, gradOutput.storage(),
            gradOutput.storageOffset() - 1 + (o - 1 + m * nOutputPlane) * outputW * outputH,
            outputH, outputW, dH, dW)
          k += 1
        }
        m += 1
      }
    }


    if (forceBatch) {
      gradOutput.squeeze(1)
    }

    if (null != wRegularizer && scaleW != 0) {
      wRegularizer.accRegularization(weight, gradWeight, scaleW)
    }
    if (null != bRegularizer && scaleB != 0) {
      bRegularizer.accRegularization(bias, gradBias, scaleB)
    }
  }

  override def parameters(): (Array[Tensor[T]], Array[Tensor[T]]) = {
    (Array(this.weight, this.bias), Array(this.gradWeight, this.gradBias))
  }

  def decayParameters(decay: T): Unit = {
    weight.apply1(ev.minus(_, decay))
    bias.apply1(ev.minus(_, decay))
  }
}

object SpatialConvolutionMap {

  def apply[@specialized(Float, Double) T: ClassTag](
    connTable: Tensor[T],
    kW: Int,
    kH: Int,
    dW: Int = 1,
    dH: Int = 1,
    padW: Int = 0,
    padH: Int = 0,
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null
  )(implicit ev: TensorNumeric[T]) : SpatialConvolutionMap[T] = {
    new SpatialConvolutionMap[T](connTable, kW, kH, dW, dH, padW, padH,
      wRegularizer, bRegularizer)
  }

  def full[@specialized(Float, Double) T: ClassTag](nin: Int, nout: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val ft = Tensor[T](nin * nout, 2)
    var p = 1
    var j = 1
    while (j <= nout) {
      var i = 1
      while (i <= nin) {
        ft.setValue(p, 1, ev.fromType[Int](i))
        ft.setValue(p, 2, ev.fromType[Int](j))
        p = p + 1
        i = i + 1
      }
      j = j + 1
    }
    ft
  }

  def oneToOne[@specialized(Float, Double) T: ClassTag](nfeat: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val ft = Tensor[T](nfeat, 2)
    var i = 1
    while (i <= nfeat) {
      ft(i)(1) = ev.fromType[Int](i)
      ft(i)(2) = ev.fromType[Int](i)
      i = i + 1
    }
    ft
  }

  def random[@specialized(Float, Double) T: ClassTag](nin: Int, nout: Int, nto: Int)(
    implicit ev: TensorNumeric[T]): Tensor[T] = {
    val nker = nto * nout
    val tbl = Tensor[T](nker, 2)
    val fi = Tensor.randperm[T](nin)
    var frcntr = 1
    val nfi = Math.floor(nin / nto).toInt // number of distinct nto chunks
    val totbl = tbl.select(2, 2)
    val frtbl = tbl.select(2, 1)
    val fitbl = fi.narrow(1, 1, nfi * nto) // part of fi that covers distinct chunks
    val ufrtbl = frtbl.unfold(1, nto, nto)
    val utotbl = totbl.unfold(1, nto, nto)
    val ufitbl = fitbl.unfold(1, nto, nto)

    // start filling frtbl
    var i = 1
    while (i <= nout) {
      // fro each unit in target map
      ufrtbl.select(1, i).copy(ufitbl.select(1, frcntr))
      frcntr = frcntr + 1
      if (frcntr - 1 == nfi) {
        fi.copy(Tensor.randperm[T](nin))
        frcntr = 1
      }
      i = i + 1
    }
    var tocntr = 1
    while (tocntr <= utotbl.size(1)) {
      utotbl.select(1, tocntr).fill(ev.fromType[Int](tocntr))
      tocntr = tocntr + 1
    }

    tbl
  }
}
