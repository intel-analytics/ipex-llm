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

import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import com.intel.analytics.bigdl.tensor.Tensor

trait InferenceSupportive {

  val logger = LoggerFactory.getLogger(getClass)

  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    logger.info(s"$name time elapsed [${cost / 1000} s, ${cost % 1000} ms].")
    result
  }

  def transferInferenceInputToTensor(input: java.util.List[java.util.List[java.lang.Float]]):
    Tensor[Float] = {
    val arrays = input.asScala.map(_.asScala.toArray.map(_.asInstanceOf[Float]))
    val _buffer = Tensor[Float]()
    arrays.length match {
      case 0 => ;
      case 1 => val size = arrays.head.length
        _buffer.resize(size)
        System.arraycopy(arrays.head, 0, _buffer.storage().array(), 0, size)
      case _ =>
        val size = arrays.head.length
        arrays.map(arr => require(size == arr.length, "input array have different lengths"))
        _buffer.resize(Array(arrays.length, size))
        var d = 0
        while (d < arrays.length) {
          System.arraycopy(arrays(d), 0, _buffer.storage().array(), d * size, size)
          d += 1
        }
    }
    _buffer
  }

}
