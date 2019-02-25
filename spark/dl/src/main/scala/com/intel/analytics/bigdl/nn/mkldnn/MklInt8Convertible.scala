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

import scala.collection.mutable.ArrayBuffer

/**
* Trait which provides MKL-DNN functionality to convert FP32 model to INT8 model
*/
trait MklInt8Convertible {
  // input dimension mask
  protected var inDimMask: Int = 0
  // output dimension mask
  protected var outDimMask: Int = 0
  // input scales
  private[mkldnn] var inScalesBuffer: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]
  // output scales
  private[mkldnn] var outScalesBuffer: ArrayBuffer[Array[Float]] = ArrayBuffer.empty[Array[Float]]


/**
* Get dimension mask of input
* @return inDimMask field which stores value of input dimension mask
*/
  def getInputDimMask(): Int = {
    inDimMask
  }

/**
* Set dimension mask of input
* @param mask value of input dimension mask to be set
* @return Unit
*/
  def setInputDimMask(mask: Int) : Unit = {
    inDimMask = mask
  }

/**
* Get dimension mask of output
* @return outDimMask field which stores value of output dimension mask
*/
  def getOutputDimMask(): Int = {
    outDimMask
  }

/**
* Set dimension mask of output
* @param mask value of output dimension mask to be set
* @return Unit
*/
  def setOutputDimMask(mask: Int): Unit = {
    outDimMask = mask
  }

/**
* Get input scales
* @return field which stores value of input scales
*/
  def getInputScales(): Array[Array[Float]] = {
    inScalesBuffer.toArray
  }

/**
* Set input scales
* Clear existing buffer of input scales, and place updated scales into the cleared buffer
* @param inScales value of input scales to be set
* @return Unit
*/
  def setInputScales(inScales: Array[Array[Float]]): Unit = {
    inScalesBuffer.clear()
    inScales.foreach(appendInputScales)
  }

/**
* Get output scales
* @return field which stores value of output scales
*/
  def getOutputScales(): Array[Array[Float]] = {
    outScalesBuffer.toArray
  }

/**
* Set output scales
* Clear existing buffer of output scales, and place updated scales into the cleared buffer
* @param outScales value of output scales to be set
* @return Unit
*/
  def setOutputScales(outScales: Array[Array[Float]]): Unit = {
    outScalesBuffer.clear()
    outScales.foreach(appendOutputScales)
  }

/**
* Append a scale, an array of float, into input scales buffer
* @param scale value of an input scale to be appended
* @return Unit
*/
  private def appendInputScales(scale: Array[Float]): Unit = {
    inScalesBuffer.append(scale)
  }

/**
* Append a scale, an array of float, into output scales buffer
* @param scale value of an output scale to be appended
* @return Unit
*/
  private def appendOutputScales(scale: Array[Float]): Unit = {
    outScalesBuffer.append(scale)
  }

/**
* Update input scales at specific index with provided new scale
* @param scale the new scale
* @param index the index of which the scale need to be updated
* @return Unit
*/
  def updateInputScales(scale: Array[Float], index: Int): Unit = {
    updateScalesHelper(inScalesBuffer, scale, index)
  }

/**
* Update output scales at specific index with provided new scale
* @param scale the new scale
* @param index the index of which the scale need to be updated
* @return Unit
*/
  def updateOutputSclaes(scale: Array[Float], index: Int): Unit = {
    updateScalesHelper(outScalesBuffer, scale, index)
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
