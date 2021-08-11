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

package com.intel.analytics.bigdl.nn.mkldnn

import com.intel.analytics.bigdl.mkl.Memory
import com.intel.analytics.bigdl.nn.MklInt8Convertible
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.mkldnn.Phase.InferencePhase
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

private[bigdl] object Utils {
  def copyMaskAndScales(from: MemoryData, to: MemoryData): Unit = {
    // here we check the from and to, they should not be null ideally
    // but if the model is mixin with blas layer, it may be null
    if (from != null && to != null && to.scales.isEmpty) {
      to.setScales(from.scales.clone())
      to.setMask(from.mask)
    }
  }

  def copyMaskAndScales(from: Array[MemoryData], to: Array[MemoryData]): Unit = {
    // here we check the from and to, they should not be null ideally
    // but if the model is mixin with blas layer, it may be null
    if (from == null || to == null) return

    val valid = (from.length == 1 || to.length == 1) || // the ConcatTable or JoinTable
      (from.length == to.length) // the same length of from and to.

    // if from has scales but to has no, copy them
    val needCopy = from.ne(to) && from.forall(_.scales.nonEmpty) && to.forall(_.scales.isEmpty)

    if (valid && needCopy) {
      if (from.length == to.length) {
        to.zip(from).foreach(x => if (x._1.scales.isEmpty) {
          x._1.setScales(x._2.scales)
          x._1.setMask(x._2.mask)
        })
      } else if (to.length == 1) {
        to.head.setScales(from.map(_.scales).transpose.map(_.max))
        require(from.map(_.mask).distinct.length == 1, s"only support the same mask")
        to.head.setMask(from.map(_.mask).distinct.head)
      } else if (to.length > 1) {
        to.foreach(_.setScales(from.head.scales))
        to.foreach(_.setMask(from.head.mask))
      }
    }
  }

  def getDefaultFormat(memoryData: MemoryData, isInOrOut: Boolean = true): Int = {
    memoryData.shape.length match {
      case 2 => if (isInOrOut) Memory.Format.nc else Memory.Format.oi
      case 4 => if (isInOrOut) Memory.Format.nchw else Memory.Format.oihw
      case _ => throw new UnsupportedOperationException("Linear only supports 2-D or 4-D")
    }
  }

  private def denseTensor(format: MemoryData, tensor: Tensor[Float],
    isInOrOut: Boolean = true, runtime: MklDnnRuntime): Tensor[Float] = {
    val reorder = ReorderMemory(HeapData(format.shape, getDefaultFormat(format, isInOrOut)))
    reorder.setRuntime(runtime)
    reorder.initFwdPrimitives(Array(format), InferencePhase)
    reorder.forward(tensor).toTensor[Float]
  }

  private def denseActivity(formats: Array[MemoryData], activity: Activity,
    isInOrOut: Boolean = true, runtime: MklDnnRuntime): Activity = {
    val ret = if (formats.length > 1) { // table
      require(formats.length == activity.toTable.length(),
        s"formats should be the same as activity")
      val table = T()

      var i = 1
      while (i <= formats.length) {
        val format = formats(i - 1)
        val tensor = activity.toTable.get[Tensor[Float]](i).get
        table(i) = denseTensor(format, tensor, isInOrOut, runtime)
        i += 1
      }

      table
    } else { // tensor
      denseTensor(formats(0), activity.toTensor[Float], isInOrOut, runtime)
    }

    ret
  }

  def getDenseIn(module: MklInt8Convertible, input: Activity): Activity = {
    if (module.isInstanceOf[MklDnnModule]) {
      val mklDnnLayer = module.asInstanceOf[MklDnnModule]
      Utils.denseActivity(mklDnnLayer.inputFormats(), input, true, mklDnnLayer.getRuntime)
    } else {
      input
    }
  }

  def getDenseOut(module: MklInt8Convertible, output: Activity): Activity = {
    if (module.isInstanceOf[MklDnnModule]) {
      val mklDnnLayer = module.asInstanceOf[MklDnnModule]
      Utils.denseActivity(mklDnnLayer.outputFormats(), output, true, mklDnnLayer.getRuntime)
    } else {
      output
    }
  }

  private def setConvNegativeInput(module: MklInt8Convertible, input: Activity): Unit = {
    if (module.isInstanceOf[SpatialConvolution]) {
      val conv = module.asInstanceOf[SpatialConvolution]
      val denseIn = getDenseIn(module, input)
      val min = denseIn.toTensor[Float].min()
      if (min >= 0.0f) {
        conv.negativeInput = false
      }
    }
  }

  def calcScales(module: AbstractModule[_, _, _], input: Activity): Unit = {
    module match {
      case mkldnnModule: MklInt8Convertible =>
        mkldnnModule.calcScales(input)
        Utils.setConvNegativeInput(mkldnnModule, input)
      case _ =>
    }
  }

  def getOutput(module: AbstractModule[_, _, _], input: Activity): Activity = {
    module match {
      case mklDnnModule: MklDnnModule => module.output.asInstanceOf[Activity]
      case _ => module.output.asInstanceOf[Activity]
    }
  }
}
