package com.intel.analytics.zoo.preprocess

import java.lang.{Float => JFloat, Integer => JInt}
import java.util.{List => JList}

import com.intel.analytics.zoo.pipeline.inference.JTensor

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

case class GloveTextProcessor(gloveFilePath: String, stopWordsCount: Int, sequenceLength: Int) extends TextProcessing {
  val embMap = doLoadEmbedding(gloveFilePath)

  override def doPreprocess(text: String, stopWordsCount: Int, sequenceLength: Int): List[List[Float]] = {
    val tokens = doTokenize(text)
    val shapedTokens = doShaping(doStopWords(tokens, stopWordsCount), sequenceLength)
    val vectorizedTokens = doVectorize(shapedTokens, embMap)
    vectorizedTokens
  }
}

//fot test
object preprocessor {
  def main(args: Array[String]): Unit = {
    val textPreprocessor = GloveTextProcessor(sys.env("EMBEDDING_PATH"), 1, 500)
    val text = "It is for for for for test test exwqwq"
    val result = textPreprocessor.doPreprocess(text, textPreprocessor.stopWordsCount, textPreprocessor.sequenceLength)
    val tempArray = ArrayBuffer[JList[JFloat]]()
    for (tempList <- result) {
      val javaList = new ArrayBuffer[JFloat](tempList.size)
      for (tempFloat <- tempList) {
        javaList.add(tempFloat.asInstanceOf[JFloat])
      }
      //System.arraycopy(tempList, 0 , javaList , 0 , tempList.size)
      tempArray.add(javaList.toList.asJava)
    }
    val input = tempArray.toArray.toList.asJava
    val data = input.flatten
    val shape = List(input.size().asInstanceOf[JInt], input.get(0).length.asInstanceOf[JInt])
    val tensorInput = new JTensor(data.asJava, shape.asJava)
    print(tensorInput)
  }
}
