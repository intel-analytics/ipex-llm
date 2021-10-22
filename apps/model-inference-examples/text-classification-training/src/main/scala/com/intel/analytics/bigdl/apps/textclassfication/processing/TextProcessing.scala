package com.intel.analytics.bigdl.apps.textclassfication.processing

import java.io.File

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.slf4j.LoggerFactory

import scala.collection.immutable.ListMap
import scala.collection.mutable.{Map => MutableMap}
import scala.io.Source

trait TextProcessing {

  val logger = LoggerFactory.getLogger(getClass)

  def doTokenize(text: String, toLowerCase: Boolean = true): List[String] = {
    val replaced = text.replaceAll("[^a-zA-Z]", " ")
    val lowerCased = toLowerCase match {
      case true => replaced.toLowerCase()
      case false => replaced
    }
    lowerCased.split("\\s+").filter(_.size > 2).toList
  }

  def doStopWords(tokens: List[String], stopWords: List[String]): List[String] = {
    tokens.filter(!stopWords.contains(_))
  }

  def doStopWords(tokens: List[String], stopWordsCount: Int): List[String] = {
    val topK = ListMap(tokens.groupBy(identity).mapValues(_.length).toSeq.sortBy(-_._2): _*).take(stopWordsCount)
    tokens.filter(!topK.keySet.contains(_))
  }

  def doShaping(tokens: List[String], sequenceLength: Int, truncateFromRight: Boolean = true): List[String] = {
    tokens.length > sequenceLength match {
      case true => {
        truncateFromRight match {
          case true => tokens.take(sequenceLength)
          case false => tokens.takeRight(sequenceLength)
        }
      }
      case false => tokens ++ List.fill[String](sequenceLength - tokens.length)("")
    }
  }

  def doIndex(tokens: List[String], wordToIndexMap: MutableMap[String, Int]): List[Int] = {
    tokens.map(token => {
      wordToIndexMap.getOrElse(token, 0)
    })
  }

  def doLoadWordToIndexMap(embeddingFile: File): MutableMap[String, Int] = {
    val source = Source.fromFile(embeddingFile, "ISO-8859-1")
    val lines = try source.getLines().toList finally source.close()
    val wordToIndexMap = MutableMap[String, Int]()
    var index = 1
    lines.map(line => {
      val values = line.split(" ")
      val word = values.head
      wordToIndexMap.put(word, index)
      index += 1
    })
    wordToIndexMap
  }

  def doPreprocess(text: String, stopWordsCount: Int, sequenceLength: Int, wordToIndexdMap: MutableMap[String, Int]): Tensor[Float] = {
    val tokens = doTokenize(text)
    val stoppedTokens = doStopWords(tokens, stopWordsCount)
    val shapedTokens = doShaping(stoppedTokens, sequenceLength)
    val indexes = doIndex(shapedTokens, wordToIndexdMap)
    Tensor(indexes.map(_.asInstanceOf[Float]).toArray, Array(sequenceLength))
  }

  def doPreprocess(text: String, stopWordsCount: Int, sequenceLength: Int, embeddingFile: File): Tensor[Float] = {
    val wordToIndexdMap = doLoadWordToIndexMap(embeddingFile)
    doPreprocess(text, stopWordsCount, sequenceLength, wordToIndexdMap)
  }

}
