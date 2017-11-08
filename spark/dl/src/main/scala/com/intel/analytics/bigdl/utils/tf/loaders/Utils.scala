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
package com.intel.analytics.bigdl.utils.tf.loaders

import java.nio.ByteOrder
import java.nio.charset.Charset
import java.util

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.tf.Context
import com.intel.analytics.bigdl.utils.tf.TensorflowToBigDL.toTensor
import org.tensorflow.framework.{AttrValue, DataType, NodeDef}

import scala.reflect.ClassTag
import collection.JavaConverters._

object Utils {
  private[loaders] def getOrSetTensor[T: ClassTag](
    node: NodeDef, context: Context[T], byteOrder: ByteOrder,
    trans: Option[Seq[(Int, Int)]] = None)(
    implicit ev: TensorNumeric[T]): (Tensor[T], Tensor[T]) = {

    if (context.containsTensor(node.getName)) {
      val result = context(node.getName)
      (result._1, result._2)
    } else {
      var weight = toTensor[T](node.getAttrMap.get("value").getTensor, byteOrder)
      trans match {
        case Some(transposes) =>
          for ((first, second) <- transposes) {
            weight = weight.transpose(first, second)
          }
          weight = weight.contiguous()
        case _ =>
      }
      val gradient = Tensor[T](weight.size())
      context.putTensor(node.getName, (weight, gradient, trans))
      (weight, gradient)
    }
  }

  private[loaders] def getString(attrMap: util.Map[String, AttrValue], key: String): String = {
    attrMap.get(key).getS.toString(Charset.defaultCharset())
  }

  private[loaders] def getInt(attrMap: util.Map[String, AttrValue], key: String): Int = {
    attrMap.get(key).getI.toInt
  }

  private[loaders] def getFloat(attrMap: util.Map[String, AttrValue], key: String): Float = {
    attrMap.get(key).getF
  }

  private[loaders] def getBoolean(attrMap: util.Map[String, AttrValue], key: String): Boolean = {
    attrMap.get(key).getB
  }

  private[loaders] def getIntList(attrMap: util.Map[String, AttrValue], key: String): Seq[Int] = {
    attrMap.get(key).getList.getIList.asScala.map(_.toInt)
  }

  private[loaders] def getType(attrMap: util.Map[String, AttrValue], key: String): DataType = {
    attrMap.get(key).getType
  }

  private[loaders] def toArray[T: ClassTag](tensor: Tensor[T]): Array[T] = {
    require(tensor.nDimension() == 1, "require 1D tensor")
    val array = new Array[T](tensor.nElement())
    var i = 0
    while(i < array.length) {
      array(i) = tensor.valueAt(i + 1)
      i += 1
    }
    array
  }
}
