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

package com.intel.analytics.bigdl.models.lm

import scopt.OptionParser

import scala.collection.mutable.ListBuffer
import scala.io.Source
import scala.util.Random

object Utils {
  case class TrainParams(
    folder: String = "/path/to/IMDB_data/",
    ngramRange: Int = 1,
    maxFeatures: Int = 20000,
    maxLen: Int = 400,
    batchSize: Int = 32,
    hiddenSize: Int = 50,
    maxEpoch: Int = 5,
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2))

  val trainParser = new OptionParser[TrainParams]("BigDL FastText Classification Example") {
    opt[String]('f', "folder")
      .text("where you put the IMDB data")
      .action((x, c) => c.copy(folder = x))

    opt[Int]('r', "ngram")
      .text("n-gram range, default is unigram")
      .action((x, c) => c.copy(ngramRange = x))

    opt[Int]('d', "maxFeatures")
      .text("max features size")
      .action((x, c) => c.copy(maxFeatures = x))

    opt[Int]('s', "maxLen")
      .text("filtering for sentences shorter than maxLen")
      .action((x, c) => c.copy(maxLen = x))

    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[Int]('h', "hidden")
      .text("hidden size")
      .action((x, c) => c.copy(hiddenSize = x))

    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  /**
    * Loads IMDB dataset.
    *
    * @param path where to store the data.
    * @param dictionarySize max number of words to include. Words are ranked
    *                       by how often they occur (in the training set) and
    *                       only the most frequent words are kept.
    * @param skipTop skip the top N most frequently occuring words
    *                (which may not be informative).
    * @param maxLen truncate sequences after this length.
    * @param startChar The start of a sequence will be marked with this character.
    *                  Set to 1 because 0 is usually the padding character.
    * @param oovChar words that were cut out because of the `dictionarySize`
    *                or `skipTop` limit will be replaced with this character.
    * @param indexFrom index actual words with this index and higher.
    * @param seed random seed for sample shuffling.
    * @return
    */
  private[bigdl] def loadData(
    path: String,
    dictionarySize: Option[Int] = None,
    skipTop: Int = 0,
    maxLen: Option[Int] = None,
    startChar: Option[Int] = Some(1),
    oovChar: Option[Int] = Some(2),
    indexFrom: Option[Int] = Some(3),
    seed: Int = 123)
  : (List[List[Int]], List[Int], List[List[Int]], List[Int]) = {
    var train = Source.fromFile(path + "/train.txt").getLines().toList
      .map(line => line.drop(1).dropRight(1).split(", ").map(_.toInt)).map(_.toList)
    var labelsTrain = Source.fromFile(path + "/train_label.txt")
      .getLines().toList.map(_.toInt)

    var test = Source.fromFile(path + "/test.txt").getLines().toList
      .map(line => line.drop(1).dropRight(1).split(", ").map(_.toInt)).map(_.toList)
    var labelsTest = Source.fromFile(path + "/test_label.txt")
      .getLines().toList.map(_.toInt)

    Random.setSeed(seed)
    train = Random.shuffle(train)
    Random.setSeed(seed)
    labelsTrain = Random.shuffle(labelsTrain)

    Random.setSeed(seed * 2)
    test = Random.shuffle(test)
    Random.setSeed(seed * 2)
    labelsTest = Random.shuffle(labelsTest)

    var x = train ++ test
    var labels = labelsTrain ++ labelsTest

    if (!startChar.isEmpty) {
      x = x.map(line => startChar.get :: line.map(i => i + indexFrom.get))
    } else if (!indexFrom.isEmpty) {
      x = x.map(_.map(i => i + indexFrom.get))
    }

    if (!maxLen.isEmpty) {
      var newX = List[List[Int]]()
      val newL = List[Int]()
      (x zip labels).map { p =>
        if (p._1.length < maxLen.get) {
          newX ::= p._1
          newL :+ p._2
        }
      }
      x = newX
      labels = newL
    }
    if (x.isEmpty) {
      throw new RuntimeException("After filtering for sequences shorter than maxlen=" +
        s"${maxLen}, no sequence was kept. \nIncrease maxlen.")
    }
    val dictSize = dictionarySize.getOrElse(x.map(line => line.max).max)

    // by convention, use 2 as OOV word
    // reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    x = x.map(line => line.map { index =>
      if (index < dictSize && index >= skipTop) index
      else if (!oovChar.isEmpty) oovChar.get
    }).asInstanceOf[List[List[Int]]]

    // TODO: When 'maxLen' option works, check whether this kind split is reasonable
    train = x.splitAt(train.length)._1
    test = x.splitAt(train.length)._2
    labelsTrain = labels.splitAt(train.length)._1
    labelsTest = labels.splitAt(train.length)._2

    (train, labelsTrain, test, labelsTest)
  }

  import Direction._

  /**
    * Pads each sequence to the same length: the length of the longest
    * sequence. If maxlen is provided, any sequence longer, than maxlen
    * is truncated to maxlen. Truncation happens off either the beginning
    * (default) or the end of the sequence.
    *
    * Supports post-padding and pre-padding (default).
    *
    * @param sequences list of lists where each element is a sequence
    * @param maxLen int, maximum length
    * @param padding pre' or 'post', pad either before or after each sequence.
    * @param truncating 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
    * @param value int, value to pad the sequences to the desired value
    * @return list of lists with dimensions (number_of_sequences, maxlen)
    */
  private[bigdl] def padSequence(
    sequences: List[List[Int]],
    maxLen: Option[Int] = None,
    padding: Option[Direction] = Some(Direction.Pre),
    truncating: Option[Direction] = Some(Direction.Pre),
    value: Option[Int] = Some(0))
  : List[List[Int]] = {
    val _maxLen = maxLen.getOrElse(sequences.map(_.length).max)
    sequences.filter(_.length > 0).zipWithIndex.map { case (sample, index) =>
      val trunc: List[Int] = if (truncating.get == Direction.Pre) {
        sample.takeRight(_maxLen)
      } else {
        sample.take(_maxLen)
      }

      if (padding == Direction.Post) {
        trunc.padTo(_maxLen, value.get)
      } else {
        List.fill(_maxLen - trunc.length)(value.get) ++ trunc
      }
    }
  }

  /**
    * Extract a *set* of n-grams from a list of integers.
    *
    * @param sample
    * @param ngram
    * @return
    */
  private[bigdl] def createNgramSet(sample: List[Int], ngram: Int = 2)
  : Set[List[Int]] = {
    sample.sliding(ngram).toSet
  }

  /**
    * Augment the input list of list (sequences) by appending n-grams values.
    *
    * @param samples
    * @param token2Index
    * @param ngramRange
    * @return
    */
  private[bigdl] def addNgram(
    samples: List[List[Int]],
    token2Index: Map[List[Int], Int],
    ngramRange: Int = 2)
  : List[List[Int]] = {
    samples.map { sample =>
      val sampleBuff = sample.to[ListBuffer]
      for (i <- 0 to (sample.length - ngramRange)) {
        for (ngram_v <- 2 to ngramRange) {
          val ngram = sample.slice(i, i + ngram_v)
          if (token2Index.contains(ngram)) {
            sampleBuff.append(token2Index(ngram))
          }
        }
      }
      sampleBuff.toList
    }
  }
}

object Direction extends Enumeration {
  type Direction = Value
  val Pre = Value("Pre")
  val Post = Value("Post")
}