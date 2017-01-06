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

import scala.collection.Iterator
import scala.collection.mutable.ArrayBuffer

 /**
  * Transformer that tokenizes a Document (article)
  * into a Seq[Seq[String]] by using Stanford Tokenizer.
  *
  */

class DocumentTokenizer() extends Transformer[String, Array[Array[String]]] {
  import edu.stanford.nlp.process.DocumentPreprocessor
  var dp: DocumentPreprocessor = null
  override def apply(prev: Iterator[String]): Iterator[Array[Array[String]]] =
    prev.map(x => {
      dp = new DocumentPreprocessor(x)
      val iterator = dp.iterator()
      val sentences = ArrayBuffer[Array[String]]()
      while (iterator.hasNext) {
        val sentence = iterator.next()
        val tokens = ArrayBuffer[String]()
        val inIter = sentence.iterator()
        while (inIter.hasNext) {
          val token = inIter.next().word()
          tokens.append(token)
        }
        sentences.append(tokens.toArray)
      }
      sentences.toArray
    })
}

object DocumentTokenizer {
  def apply(): DocumentTokenizer = new DocumentTokenizer()
}
