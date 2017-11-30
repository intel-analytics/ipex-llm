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
package com.intel.analytics.bigdl.example.chatbot

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.text.utils.SentenceToken
import com.intel.analytics.bigdl.dataset.text.{SentenceTokenizer, _}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.ml.feature
import org.apache.spark.rdd.RDD

import scala.collection.Iterator
import scala.io.Source
import scala.reflect.ClassTag

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)

  import Utils._

  val logger = Logger.getLogger(getClass)

  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
//
//      val conf = Engine.createSparkConf()
//        .setAppName("Train rnn on text")
//        .set("spark.task.maxFailures", "1")
//        .set("spark.driver.maxResultSize", "3g")
//      val sc = new SparkContext(conf)
//      Engine.init


      val dictionary = Dictionary("/home/yao/Documents/bigdl/chatbot/data/")
      val padding = "###"
      val padId = dictionary.getIndex(padding) + 1
      val padFeature = Tensor[Float](T(padId))
      val padLabel = Tensor[Float](T(-1))

      val model = if (param.modelSnapshot.isDefined) {
        Module.load[Float](param.modelSnapshot.get)
      } else {
        throw new RuntimeException("")
      }

      val seeds = Array("happy birthday have a nice day",
        "donald trump won last nights presidential debate according to snap online polls")

      var i = 0
      while (i < param.nEpochs) {
        for (seed <- seeds) {
          println("Query> " + seed)
          val evenToken = SentenceTokenizer().apply(Array(seed).toIterator).toArray
          val oddToken = (SentenceBiPadding() -> SentenceTokenizer())
            .apply(Array("").toIterator).toArray
          val labeledChat = evenToken.zip(oddToken)
            .map(chatToLabeledChat(dictionary, _)).apply(0)

          val sent1 = Tensor(Storage(labeledChat._1), 1, Array(1, labeledChat._1.length))
          var sent2 = Tensor(Storage(labeledChat._2), 1, Array(1, labeledChat._2.length))
          val timeDim = 2
          val featDim = 3
          val concat = Tensor[Float]()
          var curInput = sent2
          val end = dictionary.getIndex(SentenceToken.end) + 1
          var break = false
          model.evaluate()

          var i = 0
          // Iteratively output predicted words
          while (i < 30 && !break) {
            val output = model.forward(T(sent1, curInput)).toTensor[Float]
            val predict = output.max(featDim)._2
              .select(timeDim, output.size(timeDim)).valueAt(1, 1).toInt
            if (predict == end) break = true
            if (!break) {
              concat.resize(1, curInput.size(timeDim) + 1)
              concat.narrow(timeDim, 1, curInput.size(timeDim)).copy(curInput)
              concat.setValue(1, concat.size(timeDim), predict)
              curInput.resizeAs(concat).copy(concat)
            }
            i += 1
          }
          val predArray = new Array[Float](curInput.nElement())
          Array.copy(curInput.storage().array(), curInput.storageOffset() - 1,
            predArray, 0, curInput.nElement())
          val result = predArray.grouped(curInput.size(timeDim)).toArray[Array[Float]]
            .map(x => x.map(t => dictionary.getWord(t - 1)))
          println(result.map(x => x.mkString(" ")).mkString("\n"))
        }
      }
//      sc.stop()
    })
  }


  def chatToLabeledChat[T: ClassTag](
    dictionary: Dictionary,
    chat: (Array[String], Array[String]))(implicit ev: TensorNumeric[T])
  : (Array[T], Array[T], Array[T]) = {
    val (indices1, indices2) =
      (chat._1.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)),
        chat._2.map(x => ev.fromType[Int](dictionary.getIndex(x) + 1)))
    val label = indices2.drop(1)
    (indices1, indices2.take(indices2.length - 1), label)
  }

  def labeledChatToSample[T: ClassTag](
    labeledChat: (Array[T], Array[T], Array[T]))
    (implicit ev: TensorNumeric[T]): Sample[T] = {

    val sentence1: Tensor[T] = Tensor(Storage(labeledChat._1))
    val sentence2: Tensor[T] = Tensor(Storage(labeledChat._2))
    val label: Tensor[T] = Tensor(Storage(labeledChat._3))

    Sample(featureTensors = Array(sentence1, sentence2), labelTensor = label)
  }
}
