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

package com.intel.analytics.bigdl.dataset.text

import java.io.{File, PrintWriter, Serializable}

import org.apache.log4j.Logger
import org.apache.log4j.spi.LoggerFactory
import org.apache.spark.rdd.RDD

import scala.util.Random

/**
 * Class that help build a dictionary
 * either from tokenized text or from saved dictionary
 *
 */
class Dictionary()
  extends Serializable {

  /**
   * The length of the vocabulary
   */
  def getVocabSize(): Int = _vocabSize

  /**
   * Selected words with top-k frequencies and discarded the remaining words.
   * Return the length of the discarded words.
   */
  def getDiscardSize(): Int = _discardSize

  /**
   * Word encoding by its index in the dictionary
   */
  def word2Index(): Map[String, Int] = _word2index

  def index2Word(): Map[Int, String] = _index2word

  /**
   * Return the array of all selected words.
   */
  def vocabulary(): Array[String] = _vocabulary.toArray

  /**
   * Return the array of all discarded words.
   */
  def discardVocab(): Array[String] = _discardVocab.toArray

  /**
   * return the encoding number of a word,
   * if word does not existed in the dictionary,
   * it will return the dictionary length as the default index.
   * @param word
   */
  def getIndex(word: String): Int = {
    _word2index.getOrElse(word, _vocabSize)
  }

  def getWord(index: Float): String = {
    getWord(index.toInt)
  }

  def getWord(index: Double): String = {
    getWord(index.toInt)
  }

  /**
   * return the word with regard to the index,
   * if index is out of boundary, it will randomly
   * return a word in the discarded word list.
   * If discard word list is Empty, it will randomly
   * return a word in the existed dictionary.
   * @param index
   */
  def getWord(index: Int): String = {
    _index2word.getOrElse(index,
      if (_discardSize > 0) _discardVocab(Random.nextInt(_discardSize))
      else getWord(Random.nextInt(_vocabSize)))
  }

  /**
   * print word-to-index dictionary
   */
  def print(): Unit = {
    _word2index.foreach(x =>
      logger.info(x._1 + " -> " + x._2))
  }

  /**
   * print discard dictionary
   */
  def printDiscard(): Unit = {
    _discardVocab.foreach(x =>
      logger.info(x))
  }

  /**
   * Save the dictionary, discarded words to the saveFolder
   * directory.
   * @param saveFolder
   */
  def save(saveFolder: String): Unit = {
    val saveTo = new File(saveFolder)
    require(saveTo.isDirectory,
      "Dictionary: saveFolder should be a directory.")
    new PrintWriter(saveTo.getAbsolutePath + "/dictionary.txt") {
      write(word2Index().mkString("\n")); close
    }

    new PrintWriter(saveTo.getAbsolutePath + "/discard.txt") {
      write(discardVocab().mkString("\n")); close
    }
    logger.info("save created dictionary.txt and discard.txt to" +
      s"${saveTo.getAbsolutePath}")
  }

  def this(dataset: RDD[Array[String]], vocabSize: Int) = {
    this()
    val words = dataset.flatMap(x => x)
    logger.info(s"${words.count()} words and" +
      s"${dataset.count()} sentences processed")
    val freqDict = words
      .map(w => (w, 1))
      .reduceByKey(_ + _)
      .collect().sortBy(_._2)

    update(freqDict.toSeq, vocabSize)
  }

  def this(sentences: Iterator[Array[String]],
           vocabSize: Int) = {
    this()
    val freqDict = sentences
      .flatMap(x => x)
      .foldLeft(Map.empty[String, Int]) {
        (count, word) => count + (word -> (count.getOrElse(word, 0) + 1))
      }.toSeq.sortBy(_._2)

    update(freqDict, vocabSize)
  }

  def this(words: Array[String],
           vocabSize: Int) = {
    this()
    val freqDict = words
      .foldLeft(Map.empty[String, Int]) {
        (count, word) => count + (word -> (count.getOrElse(word, 0) + 1))
      }.toSeq.sortBy(_._2)

    update(freqDict, vocabSize)
  }

  def this(sentences: Stream[Array[String]],
           vocabSize: Int) = {
    this()
    val freqDict = sentences
      .flatMap(x => x)
      .foldLeft(Map.empty[String, Int]) {
        (count, word) => count + (word -> (count.getOrElse(word, 0) + 1))
      }.toSeq.sortBy(_._2)

    update(freqDict, vocabSize)
  }

  def this(directory: String) = {
    this()

    val dictionaryFile = new File(directory, "dictionary.txt")
    require(dictionaryFile.exists() && dictionaryFile.isFile,
      "dictionaryFile does not exist or is not a File type.")

    val discardFile = new File(directory, "discard.txt")
    require(discardFile.exists() && discardFile.isFile,
      "discardFile does not exist or is not a File type.")

    import scala.io.Source
    _word2index = Source.fromFile(dictionaryFile.getAbsolutePath)
      .getLines.map(_.stripLineEnd.split("->", -1))
      .map(fields => fields(0).stripSuffix(" ") -> fields(1).stripPrefix(" ").toInt)
      .toMap[String, Int]
    _index2word = _word2index.map(x => (x._2, x._1))
    _vocabulary = _word2index.keys.toSeq
    _vocabSize = _word2index.size
    _discardVocab = Source.fromFile(discardFile.getAbsolutePath)
      .getLines().toSeq
    _discardSize = _discardVocab.length
  }

  private def update[S <: Seq[(String, Int)]](freqDict : S, vocabSize: Int) = {
    val length = math.min(vocabSize, freqDict.length)
    _vocabulary = freqDict.drop(freqDict.length - length).map(_._1)
    _vocabSize = _vocabulary.length
    _word2index = _vocabulary.zipWithIndex.toMap
    _index2word = _word2index.map(x => (x._2, x._1))
    _discardVocab = freqDict.take(freqDict.length - length).map(_._1)
    _discardSize = _discardVocab.length
  }

  @transient
  private val logger = Logger.getLogger(getClass)

  private var _vocabSize: Int = 0
  private var _discardSize: Int = 0
  private var _word2index: Map[String, Int] = null
  private var _index2word: Map[Int, String] = null
  private var _vocabulary: Seq[String] = null
  private var _discardVocab: Seq[String] = null
}

object Dictionary {
  def apply(sentences: Iterator[Array[String]], vocabSize: Int)
  : Dictionary = new Dictionary(sentences, vocabSize)

  def apply(words: Array[String], vocabSize: Int)
  : Dictionary = new Dictionary(words, vocabSize)

  def apply(dataset: Stream[Array[String]], vocabSize: Int)
  : Dictionary = new Dictionary(dataset, vocabSize)

  def apply(directory: String)
  : Dictionary = new Dictionary(directory)

  def apply(dataset: RDD[Array[String]], vocabSize: Int = 10000)
  : Dictionary = new Dictionary(dataset, vocabSize)
}
