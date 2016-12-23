/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.dataset.text

import com.intel.analytics.bigdl.dataset.Transformer
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger

import scala.collection.Iterator

object TextSeqToTensorSeq {
  val logger = Logger.getLogger(getClass)

  def apply(dictionaryLength: Int): TextSeqToTensorSeq = {
    new TextSeqToTensorSeq(dictionaryLength)
  }
}

class TextSeqToTensorSeq(dictionaryLength: Int)
  extends Transformer[(Seq[Int], Seq[Int]), (Tensor[Float], Tensor[Float])] {
  override def apply(prev: Iterator[(Seq[Int], Seq[Int])])
  : Iterator[(Tensor[Float], Tensor[Float])] = {
    prev.map(x => {
      val data = x._1
      val label = x._2

      val numOfWords = data.length
      val input = Tensor[Float](numOfWords, dictionaryLength)
      val target = Tensor[Float](numOfWords)
      var i = 0
      while (i < numOfWords) {
        input.setValue(i + 1, data(i).toString.toInt + 1, 1.0f)
        target.setValue(i + 1, label(i).toString.toInt + 1)
        i += 1
      }
      (input, target)
    })
  }
}
