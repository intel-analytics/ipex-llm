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

package com.intel.analytics.bigdl.friesian.serving.utils.recall

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Log4Error, T}
import com.intel.analytics.bigdl.friesian.serving.recall.IndexService
import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.ml.linalg.DenseVector

import java.util.{List => JList}
import scala.collection.JavaConverters._
import scala.collection.mutable

object RecallUtils {
  private val logger: Logger = LogManager.getLogger(classOf[IndexService].getName)

  def constructActivity(data: JList[Any]): Tensor[Float] = {
    Tensor[Float](T.seq(data.asScala.map {
      case d: Int => d.toFloat
      case d: Double => d.toFloat
      case d: Float => d
      case d =>
        throw new IllegalArgumentException(s"Only numeric values are supported, but got ${d}")
    }))
  }

  def featureObjToFloatArr(feature: Any): Array[Float] = {
    feature match {
      case d: Activity => activityToFloatArr(d)
      case d: Array[Any] =>
        if (d.length != 1) {
          throw new Exception(s"Feature column number should be 1, but got: ${d.length}")
        }
        d(0) match {
          case f: DenseVector => denseVectorToFloatArr(f)
          case f: mutable.WrappedArray[Any] => f.array.map(_.toString.toFloat)
          case f: Array[Float] => f
          case _ =>
            Log4Error.invalidInputError(condition = false, s"Unsupported " +
              s"user vector type, only Activity, DenseVector, WrappedArray and Float[] are " +
              s"supported, but got ${d.getClass.getName}", "")
            new Array[Float](0)
        }
      case d =>
        Log4Error.invalidInputError(condition = false, s"Unsupported " +
          s"user vector type, only Activity, DenseVector, WrappedArray and Float[] are " +
          s"supported, but got ${d.getClass.getName}", "")
        new Array[Float](0)
    }
  }

  def activityToFloatArr(data: Activity): Array[Float] = {
    val dTensor: Tensor[Float] = data.toTensor
    val result = dTensor.squeeze(1).toArray()
    result
  }

  def denseVectorToFloatArr(data: DenseVector): Array[Float] = {
    data.toArray.map(_.toFloat)
  }
}
