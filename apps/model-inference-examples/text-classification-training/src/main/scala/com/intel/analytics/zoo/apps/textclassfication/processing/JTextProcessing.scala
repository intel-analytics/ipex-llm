package com.intel.analytics.zoo.apps.textclassfication.processing

import java.io.File
import java.lang.{Float => JFloat, Integer => JInt}
import java.util.{List => JList, Map => JMap}

import com.intel.analytics.zoo.pipeline.inference.{InferenceSupportive, JTensor}

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

class JTextProcessing extends TextProcessing with InferenceSupportive {

  def tokenize(text: String): JList[String] = {
    doTokenize(text).asJava
  }

  def stopWords(tokens: JList[String], stopWordsCount: JInt): JList[String] = {
    doStopWords(tokens.asScala.toList, stopWordsCount.asInstanceOf[Int]).asJava
  }

  def stopWords(tokens: JList[String], stopWords: JList[String]): JList[String] = {
    doStopWords(tokens.asScala.toList, stopWords.asScala.toList).asJava
  }

  def shaping(tokens: JList[String], sequenceLength: JInt): JList[String] = {
    doShaping(tokens.asScala.toList, sequenceLength.asInstanceOf[Int]).asJava
  }

  def index(tokens: JList[String], wordToIndexMap: JMap[String, JInt]): JList[JFloat] = {
    val scalaWordToIndexMap = wordToIndexMap.map { case (word, index) => (word, index.asInstanceOf[Int]) }
    doIndex(tokens.asScala.toList, scalaWordToIndexMap).map(_.asInstanceOf[JFloat]).asJava
  }

  def loadWordToIndexMap(embeddingFile: File): JMap[String, JInt] = {
    doLoadWordToIndexMap(embeddingFile).map { case (word, index) => (word, index.asInstanceOf[JInt]) }.asJava
  }

  def preprocess(text: String, stopWordsCount: JInt, sequenceLength: JInt, wordToIndexMap: JMap[String, JInt]): JTensor = {
    val tensor = doPreprocess(text, stopWordsCount.asInstanceOf[Int], sequenceLength.asInstanceOf[Int],
      wordToIndexMap.map { case (word, index) => (word, index.asInstanceOf[Int]) })
    transferTensorToJTensor(tensor)
  }

  def preprocess(text: String, stopWordsCount: JInt, sequenceLength: JInt, embeddingFile: File): JTensor = {
    val tensor = doPreprocess(text, stopWordsCount.asInstanceOf[Int], sequenceLength.asInstanceOf[Int], embeddingFile)
    transferTensorToJTensor(tensor)
  }

}
