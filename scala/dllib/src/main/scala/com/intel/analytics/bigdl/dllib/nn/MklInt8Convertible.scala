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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Table

import scala.collection.mutable.ArrayBuffer


/**
 * Trait which provides MKL-DNN functionality to convert from FP32 to INT8
 */
trait MklInt8Convertible {
  // input dimension mask
  protected var inputDimMask: Int = 0
  // output dimension mask
  protected var outputDimMask: Int = 0
  // weight dimension mask
  protected var weightDimMask: Int = 0
  // input activation scales
  private[nn] var inputScalesBuffer: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]
  // output scales
  private[nn] var outputScalesBuffer: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]
  // weight scales
  private[nn] var weightScalesBuffer: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]

  /**
   * Calculate the required scales for converting int8 modules
   * Currently there are four type of modules should be supported:
   * 1) Graph: calculate scales for input and output
   * 2) Linear: calculate scales for input, output and weight
   * 3) Spatial Convolution: calculate scales for input, output and weight
   * 4) Sequential: calculate scales for input, output as well as the scales of submodules
   * 5) ConcatTable: calculate scales for input, output as well as the scales of submodules
   * @param inputActvt input activity
   */
  def calcScales(inputActvt: Activity): Unit = {

    if (inputActvt != null) {
      val module = this.asInstanceOf[AbstractModule[_, _, Float]]
      // do not forward here, because the input maybe not the real input
      // such as th ReLU(true) will do the computing inplace
      val outputActvt = module.output.asInstanceOf[Activity]

      module match {
        case graph: Graph[Float] => calcGraphScales(inputActvt, outputActvt)
        // handlers for BLAS modules
        case linear: Linear[Float@unchecked] =>
          calcModuleScales(inputActvt, outputActvt, getWeight(linear))
        case spatialConv: SpatialConvolution[Float@unchecked] =>
          calcModuleScales(inputActvt, outputActvt, getWeight(spatialConv))
        case relu: ReLU[Float@unchecked] =>
          calcModuleScales(inputActvt, outputActvt)
        case caddTable: CAddTable[Float@unchecked, Float@unchecked] =>
          calcModuleScales(inputActvt, outputActvt)
        case bn: SpatialBatchNormalization[Float@unchecked] =>
          calcModuleScales(inputActvt, outputActvt)
        case sequential: Sequential[Float@unchecked] =>
          calcSequentialScales(inputActvt, outputActvt)
        case concatTable: ConcatTable[Float@unchecked] =>
          calcConcatTableScales(inputActvt, outputActvt)
        // handlers for DNN modules
        case dnnLinear: mkldnn.Linear =>
          calcModuleScales(inputActvt, outputActvt, getWeight(dnnLinear))
        case dnnSpatialConv: mkldnn.SpatialConvolution =>
          calcModuleScales(inputActvt, outputActvt, getWeight(dnnSpatialConv))
        case dnnSequential: mkldnn.Sequential =>
          calcSequentialScales(inputActvt, outputActvt)
        case dnnConcatTable: mkldnn.ConcatTable =>
          calcConcatTableScales(inputActvt, outputActvt)
        case relu: mkldnn.ReLU =>
          calcModuleScales(inputActvt, outputActvt)
        case bn: mkldnn.SpatialBatchNormalization =>
          calcModuleScales(inputActvt, outputActvt)
        case caddTable: mkldnn.CAddTable =>
          calcModuleScales(inputActvt, outputActvt)
        case _ => throw new UnsupportedOperationException(
          "Int8 conversion is not supported for module: " + module.getName()
        )
      }
    }
  }

  private[bigdl] def flushWeightScales(weight: Tensor[Float]): Unit = {
    weightScalesBuffer.clear()
    appendWeightScales(calcTensorScale(weight, weightDimMask))
  }

  /**
   * Calculate module's scales given its input and output
   * Store calculated scales in array buffers
   * @param inputActvt input activity
   * @param outputActvt output activity
   */
  private def calcModuleScales(inputActvt: Activity, outputActvt: Activity): Unit = {
    if (inputActvt != null) {
      val denseIn = mkldnn.Utils.getDenseIn(this, inputActvt)
      calcActivityScales(denseIn, inputDimMask).foreach(appendInputScales)
    }

    if (outputActvt != null) {
      val denseOut = mkldnn.Utils.getDenseOut(this, outputActvt)
      calcActivityScales(denseOut, outputDimMask).foreach(appendOutputScales)
    }
  }

  /**
   * Calculate module's scales given its input, output and weight
   * @param inActivity input activity
   * @param outActivity output activity
   * @param weightTensor weight
   */
  private def calcModuleScales(inActivity: Activity, outActivity: Activity,
                               weightTensor: Tensor[Float]): Unit = {
    // calculate scales for input and output
    calcModuleScales(inActivity, outActivity)
    // calculate scales for weight
    appendWeightScales(calcTensorScale(weightTensor, weightDimMask))
  }

  /**
   * Calculate scales given activity and mask
   * @param activity target activity to get scales
   * @param mask dimension mask associated with target activity
   */
  private def calcActivityScales(activity: Activity, mask: Int): Array[Array[Float]] = {
    activity match {
      case tensor: Tensor[Float@unchecked] => Array(calcTensorScale(activity.toTensor[Float], mask))
      case table: Table => activity.toTable.map[Array[Float]](elem => {
          val index: Any = elem._1
          val tensor: Tensor[Float] = elem._2.asInstanceOf[Tensor[Float]]
          calcTensorScale(tensor, mask)
        }).toArray
      case _ => throw new IllegalArgumentException("Invalid activity " + activity)
    }
  }

  /** Given a tensor and a dimension mask, calculate the scales of this tensor
   * @param tensor tensor of float, stores high dimension data
   * @param mask dimension mask
   * @return scalesBuffer Array, an array stores scales
   */
  private def calcTensorScale(tensor: Tensor[Float], mask: Int): Array[Float] = {
    // we must clone the tensor, the abs will change the original tensor's value
    if (mask == 0) { // no mask performed, return max of tensor storage
      Array(tensor.clone().abs().max())
    } else if (scala.math.pow(2, tensor.dim()) - 1 == mask) {
      // mask bits are ON for all dimensions
      // return the abs value of tensor as an array
      tensor.clone().abs().storage().toArray[Float]
    } else {
      Utils.calcScales(tensor, mask)
    }
  }

  /**
   * Scales calculator for Sequential Module
   * @param inputActvt input of the Sequential Module
   * @param outputActvt output of the Sequential Module
   */
  private def calcSequentialScales(inputActvt: Activity, outputActvt: Activity): Unit = {
    require(this.isInstanceOf[Sequential[Float@unchecked]] || this.isInstanceOf[mkldnn.Sequential],
      this.getClass.getName + " is not an instance of Sequential.")

    val module: DynamicContainer[_, _, Float] = this.asInstanceOf[DynamicContainer[_, _, Float]]

    // output of previous module is the input of current module
    var prevOutputActivity: Activity = inputActvt

    // calc scales for main module
    this.calcModuleScales(inputActvt, outputActvt)

    // Iterator of Sequential modules
    val moduleIter = module.modules.iterator
    // calc scales for sub-module
    while (moduleIter.hasNext) {
      val currModule = moduleIter.next()
      if (currModule.isInstanceOf[MklInt8Convertible]) {
        val cvtbModule = currModule.asInstanceOf[MklInt8Convertible]
        cvtbModule.calcScales(prevOutputActivity)
      }
      // update previous output
      prevOutputActivity = currModule.output
    }
  }

  /**
   * Scales calculator for ConcatTable module
   * Submodules inside ConcatTable share the same input
   * @param inputActvt input of the ConcatTable Module
   * @param outputActvt output of the ConcatTable Module
   */
  private def calcConcatTableScales(inputActvt: Activity, outputActvt: Activity): Unit = {
    require(this.isInstanceOf[ConcatTable[Float@unchecked]] || this.isInstanceOf[mkldnn.ConcatTable]
      , this.getClass.getName + " is not an instance of ConcatTable.")

    val module: DynamicContainer[_, _, Float] = this.asInstanceOf[DynamicContainer[_, _, Float]]

    // calc scales for main module
    this.calcModuleScales(inputActvt, outputActvt)

    // calc scales for sub-module
    val moduleIter = module.modules.iterator
    while (moduleIter.hasNext) {
      val currModule = moduleIter.next()
      if (currModule.isInstanceOf[MklInt8Convertible]) {
        val cvtbModule = currModule.asInstanceOf[MklInt8Convertible]
        cvtbModule.calcScales(inputActvt)
      }
    }
  }

  /**
   * Scales calculator for Graph module
   * Submodules inside Graph are traversed based on its topological sort
   * The order can obtain from by calling getForwardExecutions
   * @param inputActvt input activity of the graph module
   * @param outputActvt output activity of the graph module
   */
  private def calcGraphScales(inputActvt: Activity, outputActvt: Activity): Unit = {
    require(this.isInstanceOf[Graph[Float@unchecked]], this.getClass.getName +
    " is not an instance of Graph[Float]")

    // calc scales for main module
    calcModuleScales(inputActvt, outputActvt)

    // calc scales for sub-module
    val module: Graph[Float] = this.asInstanceOf[Graph[Float]]
    val outputNodes = module.getForwardExecutions()
    var i = 0
    // traverse through all the sub-modules
    while(i < outputNodes.length) {
      // get current sub-module
      val currNode = outputNodes(i)
      // get the input activity of current sub-module
      val currInputActvt = module.findInput(currNode, inputActvt)
      // calculate scales if current sub-module is int8 convertible
      if (currNode.element.isInstanceOf[MklInt8Convertible]) {
        currNode.element.asInstanceOf[MklInt8Convertible].calcScales(currInputActvt)
      }
      i += 1
    }
  }

  /**
   * Helper function to get weight from module parameter
   * @param module the module to get weight from
   * @return a tensor contains weight
   */
  private def getWeight(module: AbstractModule[_, _, Float]): Tensor[Float] = {
    if (module != null) {
      // the getParameters will flatten the weight and bias, it's wrong
      val weight = module.parameters()._1(0)
      // If the weight is came from nn.SpatialConvolution and the nGroup is 1,
      // we need to skip the first dimension. Because if the group is 1, mkldnn thinks
      // it's 4-D tensor weight. But for original nn.SpatialConvolution, for convenience,
      // it always use 5-D tensor weight although the nGroup is 1.
      if (module.isInstanceOf[SpatialConvolution[Float]] && weight.size(1) == 1) {
        weight.select(1, 1)
      } else {
        weight
      }
    } else {
      null
    }
  }

  /**
   * Get dimension mask of input
   * @return inputDimMask field which stores value of input dimension mask
   */
  def getInputDimMask(): Int = {
    inputDimMask
  }

  /**
   * Set dimension mask of input
   * @param mask value of input dimension mask to be set
   * @param overrideSubmodules when set it to true,
   *             update mask including itself and submodules,
   *             otherwise only update mask to module itself.
   * @return Unit
   */
  def setInputDimMask(mask: Int, overrideSubmodules: Boolean = false) : Unit = {
    inputDimMask = mask
    if (this.isInstanceOf[Container[_, _, Float@unchecked]] && overrideSubmodules == true) {
      val container = this.asInstanceOf[Container[_, _, Float@unchecked]]
      val modules = container.modules
      modules.foreach(module => {
        if (module.isInstanceOf[MklInt8Convertible]) {
          module.asInstanceOf[MklInt8Convertible].setInputDimMask(mask, overrideSubmodules)
        }
      })
    }
  }

  /**
   * Get dimension mask of output
   * @return outputDimMask field which stores value of output dimension mask
   */
  private[bigdl] def getOutputDimMask(): Int = {
    outputDimMask
  }

  /**
   * Set dimension mask of output
   * @param mask value of output dimension mask to be set
   * @param overrideSubmodules when set it to true,
   *             update mask in full scope including itself and submodules,
   *             otherwise only update mask to module itself.
   * @return Unit
   */
  def setOutputDimMask(mask: Int, overrideSubmodules: Boolean = false): Unit = {
    outputDimMask = mask
    if (this.isInstanceOf[Container[_, _, Float@unchecked]] && overrideSubmodules == true) {
      val container = this.asInstanceOf[Container[_, _, Float@unchecked]]
      val modules = container.modules
      modules.foreach(module => {
        if (module.isInstanceOf[MklInt8Convertible]) {
          module.asInstanceOf[MklInt8Convertible].setOutputDimMask(mask, overrideSubmodules)
        }
      })
    }
  }

  /**
   * Get dimension mask of weight
   * @return weightDimMask which stores value of weight mask
   */
  def getWeightDimMask(): Int = {
    weightDimMask
  }

  /**
   * Set dimension mask for weight
   * @param mask value of weight mask to be set
   * @param overrideSubmodules when set it to true,
   *             update mask in full scope including itself and submodules,
   *             otherwise only update mask to module itself.
   * @return Unit
   */
  def setWeightDimMask(mask: Int, overrideSubmodules: Boolean = false): Unit = {
    weightDimMask = mask
    if (this.isInstanceOf[Container[_, _, Float@unchecked]] && overrideSubmodules == true) {
      val container = this.asInstanceOf[Container[_, _, Float@unchecked]]
      val modules = container.modules
      modules.foreach(module => {
        if (module.isInstanceOf[MklInt8Convertible]) {
          module.asInstanceOf[MklInt8Convertible].setWeightDimMask(mask, overrideSubmodules)
        }
      })
    }
  }

  /**
   * Get input scales
   * @return field which stores value of input scales
   */
  def getInputScales(): Array[Array[Float]] = {
    inputScalesBuffer.toArray
  }

  /**
   * Set input scales
   * Clear existing buffer of input scales, and place updated scales into the cleared buffer
   * @param inScales value of input scales to be set
   * @return Unit
   */
  def setInputScales(inScales: Array[Array[Float]]): Unit = {
    inputScalesBuffer.clear()
    inScales.foreach(appendInputScales)
  }

  /**
   * Get output scales
   * @return field which stores value of output scales
   */
  def getOutputScales(): Array[Array[Float]] = {
    outputScalesBuffer.toArray
  }

  /**
   * Set output scales
   * Clear existing buffer of output scales, and place updated scales into the cleared buffer
   * @param outScales value of output scales to be set
   * @return Unit
   */
  def setOutputScales(outScales: Array[Array[Float]]): Unit = {
    outputScalesBuffer.clear()
    outScales.foreach(appendOutputScales)
  }

  /**
   * Get weight scales
   * @return field which stores value of weight scales
   */
  def getWeightScales(): Array[Array[Float]] = {
    weightScalesBuffer.toArray
  }

  /**
   * Set weight scales
   * Clear existing buffer of weight scales, and place updated scales into the cleared buffer
   * @param weightScales value of weight scales to be set
   * @return Unit
   */
  def setWeightScales(weightScales: Array[Array[Float]]): Unit = {
    weightScalesBuffer.clear()
    weightScales.foreach(appendWeightScales)
  }

  /**
   * Append a scale, an array of float, into input scales buffer
   * @param scale value of an input scale to be appended
   * @return Unit
   */
  private def appendInputScales(scale: Array[Float]): Unit = {
    inputScalesBuffer.append(scale)
  }

  /**
   * Append a scale, an array of float, into output scales buffer
   * @param scale value of an output scale to be appended
   * @return Unit
   */
  private def appendOutputScales(scale: Array[Float]): Unit = {
    outputScalesBuffer.append(scale)
  }

  /**
   * Append a scale, an array of float, into weight scales buffer
   * @param scale value of an weight scale to be appended
   * @return Unit
   */
  private def appendWeightScales(scale: Array[Float]): Unit = {
    weightScalesBuffer.append(scale)
  }

  /**
   * Update input scales at specific index with provided new scale
   * @param scale the new scale
   * @param index the index of which the scale need to be updated
   * @return Unit
   */
  private def updateInputScales(scale: Array[Float], index: Int): Unit = {
    updateScalesHelper(inputScalesBuffer, scale, index)
  }

  /**
   * Update output scales at specific index with provided new scale
   * @param scale the new scale
   * @param index the index of which the scale need to be updated
   * @return Unit
   */
  private def updateOutputScales(scale: Array[Float], index: Int): Unit = {
    updateScalesHelper(outputScalesBuffer, scale, index)
  }

  /**
   * Update weight scales at specific index with provided new scale
   * @param scale the new scale
   * @param index the index of which the scale need to be updated
   * @return Unit
   */
  private def updateWeightScales(scale: Array[Float], index: Int): Unit = {
    updateScalesHelper(weightScalesBuffer, scale, index)
  }

  /**
   * Scales update helper. Replace scale at specific index with provided new scale
   * @param scales the scales arrayBuffer to be updated
   * @param scale the new scale
   * @param index the index of which the scale need to be updated
   * @return Unit
   */
  private def updateScalesHelper(scales: ArrayBuffer[Array[Float]],
                                 scale: Array[Float], index: Int): Unit = {
    if (scales.length - 1 < index) {
      scales.append(scale)
    }

    scales(index).indices.foreach(i =>
      if (scale(i) > scales(index)(i)) {
        scales(index)(i) = scale(i)
      })
  }
}
