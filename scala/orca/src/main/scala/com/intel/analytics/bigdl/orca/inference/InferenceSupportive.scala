/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.inference

import com.intel.analytics.bigdl.dataset.Sample
import org.slf4j.LoggerFactory

import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.tensor.Tensor
import java.util.{List => JList}
import java.lang.{Float => JFloat}
import java.lang.{Integer => JInt}


trait InferenceSupportive {

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    InferenceSupportive.logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  @inline
  private def product(input: JList[JInt]): Int = {
    var i = 0
    val length = input.size()
    var product = 1
    while (i < length) {
      product = product * input.get(i)
      i += 1
    }
    product
  }

  @inline
  private def toFloatArray(data: JList[JFloat]): Array[Float] = {
    val length = data.size()
    val result = new Array[Float](length)
    var i = 0
    while (i < length) {
      result(i) = data.get(i)
      i += 1
    }
    result
  }

  @inline
  private def toIntArray(data: JList[JInt]): Array[Int] = {
    val length = data.size()
    val result = new Array[Int](length)
    var i = 0
    while (i < length) {
      result(i) = data.get(i)
      i += 1
    }
    result
  }


  def transferTensorToJTensor(input: Tensor[Float]): JTensor = {
    val storageOffset = input.storageOffset - 1
    val res = new Array[Float](input.nElement())
    System.arraycopy(input.storage().array(), storageOffset, res, 0, res.length)
    val outputShape = input.size()
    new JTensor(res, outputShape, false)
  }

}

object InferenceSupportive {
  val logger = LoggerFactory.getLogger(getClass)
}
