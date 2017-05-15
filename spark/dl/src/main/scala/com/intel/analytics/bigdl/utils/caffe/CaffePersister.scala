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
package com.intel.analytics.bigdl.utils.caffe

import caffe.Caffe.LayerParameter
import com.intel.analytics.bigdl.nn.Graph
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Node
import com.intel.analytics.bigdl.utils.caffe.CaffePersister.LayerNode

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
/**
 * An utility to convert BigDL model into caffe format
 *
 * @param prototxtPath  path to store model definition file
 * @param modelPath  path to store model weights and bias
 * @param module BigDL module to be converted
 */
class CaffePersister[T: ClassTag](prototxtPath: String,
      modelPath: String,
      val module : Graph[T])(implicit ev: TensorNumeric[T]) {
  val layers : ArrayBuffer[LayerNode] = new ArrayBuffer[LayerNode]()
  def saveAsCaffe() : Unit = {

  }

  private def convertToCaffe() : Unit = {

  }

  private def copyParameters() : Unit = {

  }

  private def save() : Unit = {

  }

}

object CaffePersister{
  type LayerNode = Node[LayerParameter]
}
