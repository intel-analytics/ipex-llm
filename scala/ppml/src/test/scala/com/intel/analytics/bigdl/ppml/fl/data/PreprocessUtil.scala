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

package com.intel.analytics.bigdl.ppml.fl.data

import com.intel.analytics.bigdl.dllib.tensor.{Storage, Tensor}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer
import scala.io.Source

/**
 * This util is used to process the origin csv files with String type features
 * to numerical features. It is currently used in FGBoost scala and python validation.
 * Because process library may vary from Scala and Python so that the results could not
 * be guaranteed to be consistent. This util will preprocess data and write them to file.
 */
object PreprocessUtil {
  def preprocessing(sources: Iterator[String],
                    testSources: Iterator[String],
                    rowkeyName: String,
                    labelName: String):
  (Array[Tensor[Float]], Array[Tensor[Float]], Array[Float], Array[String]) = {
    val headers = sources.next().split(",").map(_.trim)
    val trainHeaders = headers.toBuffer
    val testHeaders = testSources.next()
    println(headers.mkString(","))
    val originData = sources.map(_.split(",").map(_.trim)).toArray
    val testOrigin = testSources.map(_.split(",").map(_.trim)).toArray
    val rowKeyIndx = headers.indexOf(rowkeyName)
    val labelIndx = headers.indexOf(labelName)
    val trainLabels = if (labelIndx > -1) {
      originData.map(l => math.log(l(labelIndx).toDouble).toFloat)
    } else {
      Array[Float]()
    }
    val data = if (labelIndx == -1) {
      trainHeaders.remove(rowKeyIndx)
      originData.map(_.toBuffer).map { a =>
        a.remove(rowKeyIndx)
        a.toArray
      }
    } else if (rowKeyIndx < labelIndx) {
      trainHeaders.remove(rowKeyIndx)
      trainHeaders.remove(labelIndx - 1)
      originData.map(_.toBuffer).map { a =>
        a.remove(rowKeyIndx)
        a.remove(labelIndx - 1)
        a.toArray
      }
    } else {
      trainHeaders.remove(labelIndx)
      trainHeaders.remove(rowKeyIndx - 1)
      originData.map(_.toBuffer).map { a =>
        a.remove(labelIndx)
        a.remove(rowKeyIndx - 1)
        a.toArray
      }
    }
    val testRowKeyIndx = testHeaders.indexOf(rowkeyName)
    val testData = testOrigin.map(_.toBuffer).map { o =>
      o.remove(testRowKeyIndx)
      o.toArray
    }

    val flattenHeader = new ArrayBuffer[String]()
    // replace NA with average for number column
    val msSubClassIndex = trainHeaders.indexOf("MSSubClass")
    val features = trainHeaders.indices.map { col =>
      // Do preprocess column-wise
      val features = data.map(_ (col))
      val testFeatures = testData.map(_ (col))
      val noneNaFeatures = features.filter(_ != "NA")
      if (noneNaFeatures.length.toFloat / features.length <= 0.6f) {
        // Too many NA features, ignore this column
        (col, -1, features.map(_ => Array[Float]()),
          testFeatures.map(_ => Array[Float]()), trainHeaders(col))
      } else {
        val isNum = if (col == msSubClassIndex) {
          // Since the numerical-like column MSSubClass is actually categorical,
          // we need to convert it from numerical to string.
          false
        } else {
          scala.util.Try(noneNaFeatures.head.toFloat).isSuccess
        }
        if (isNum) {
          // fill NA to average, an array for a column
          val avg = (noneNaFeatures.map(_.toFloat).sum / noneNaFeatures.length)
          val trainNumFeatures = features.map { d =>
            if (d == "NA") avg else d.toFloat
          }
          val testNumFeatures = testFeatures.map { t =>
            if (t == "NA") avg else t.toFloat
          }
          val std = Math.sqrt(trainNumFeatures.map(v =>
            Math.pow(v - avg, 2)).sum / trainNumFeatures.length).toFloat
          (col, 0, trainNumFeatures.map(v => Array((v - avg) / std)),
            testNumFeatures.map(v => Array((v - avg) / std)), trainHeaders(col))
        } else {
          // fill NA to max occur, an 2-d array for a column, last dim is one-hot
          val grouped = features.groupBy(v => v).mapValues(_.length)
          val fkeys = grouped.toArray.map(_._1).filter(_ != "NA").sorted
          val default = fkeys.indexOf(grouped.toArray.maxBy(_._2)._1)
          val trainCateFeatures = features.map { feature =>
            val indx = if (feature == "NA") default else fkeys.indexOf(feature)
            val oneHot = new Array[Float](fkeys.length)
            oneHot(indx) = 1f
            oneHot
          }
          val colPrefix = trainHeaders(col)
          fkeys.indices.foreach(i => flattenHeader.append(s"${colPrefix}_$i"))
          val testCateFeatures = testFeatures.map { feature =>
            val indx = if (feature == "NA") default else fkeys.indexOf(feature)
            val oneHot = new Array[Float](fkeys.length)
            if (indx != -1) oneHot(indx) = 1f
            oneHot
          }
          (col, 1, trainCateFeatures, testCateFeatures, trainHeaders(col))
        }
      }
    }
    trainHeaders.indices.foreach { col =>
      // Do preprocess column-wise
      val features = data.map(_ (col))
      val testFeatures = testData.map(_ (col))
      val noneNaFeatures = features.filter(_ != "NA")
      if (noneNaFeatures.length.toFloat / features.length <= 0.6f) {
        // Too many NA features, ignore this column
        (col, -1, features.map(_ => Array[Float]()),
          testFeatures.map(_ => Array[Float]()), trainHeaders(col))
      } else {
        val isNum = if (col == msSubClassIndex) {
          // Since the numerical-like column MSSubClass is actually categorical,
          // we need to convert it from numerical to string.
          false
        } else {
          scala.util.Try(noneNaFeatures.head.toFloat).isSuccess
        }
        if (isNum) {
          // fill NA to average, an array for a column
          flattenHeader.append(trainHeaders(col))
          val avg = (noneNaFeatures.map(_.toFloat).sum / noneNaFeatures.length)
          val trainNumFeatures = features.map { d =>
            if (d == "NA") avg else d.toFloat
          }
          val testNumFeatures = testFeatures.map { t =>
            if (t == "NA") avg else t.toFloat
          }
          val std = Math.sqrt(trainNumFeatures.map(v =>
            Math.pow(v - avg, 2)).sum / trainNumFeatures.length).toFloat
          (col, 0, trainNumFeatures.map(v => Array((v - avg) / std)),
            testNumFeatures.map(v => Array((v - avg) / std)), trainHeaders(col))
        }
      }
    }
    println("Numeric features: ")
    features.filter(_._2 == 0).foreach { f =>
      print(trainHeaders(f._1) + ", ")
    }

    println("Categorical features: ")
    features.filter(_._2 == 1).foreach { f =>
      print(trainHeaders(f._1) + ",")
    }


    val trainFeatures = data.indices.map { row =>
      val numF = features.filter(_._2 == 0).map(_._3(row))
      val cateF = features.filter(_._2 == 1).map(_._3(row))
      val f = (cateF ++ numF).toArray.flatten
      Tensor[Float](Storage[Float](f))
    }.toArray

    val testFeatures = testData.indices.map { row =>
      val numF = features.filter(_._2 == 0).map(_._4(row))
      val cateF = features.filter(_._2 == 1).map(_._4(row))
      val f = (cateF ++ numF).toArray.flatten
      Tensor[Float](Storage[Float](f))
    }.toArray
    (trainFeatures, testFeatures, trainLabels, flattenHeader.toArray)
  }
  def writePreprocessed(dataPathTrain: String,
                        dataPathTest: String,
                        rowkeyName: String = "Id",
                        labelName: String = "Label"): Unit = {
    val sources = Source.fromFile(dataPathTrain, "utf-8").getLines()
    val testSources = Source.fromFile(dataPathTest, "utf-8").getLines()

    val (trainFeatures, testFeatures, trainLabels, flattenHeader) =
      preprocessing(sources, testSources, rowkeyName, labelName)

    val fileTrain = new File(s"${dataPathTrain.split("\\.").head}-preprocessed.csv")
    val bwTrain = new BufferedWriter(new FileWriter(fileTrain))

    bwTrain.write(s"${flattenHeader.mkString(",")},$labelName\n")
    trainFeatures.indices.foreach(i => {
      bwTrain.write(s"${trainFeatures(i).storage().array().mkString(",")},${trainLabels(i)}\n")
    })
    bwTrain.close()

    val fileTest = new File(s"${dataPathTest.split("\\.").head}-preprocessed.csv")
    val bwTest = new BufferedWriter(new FileWriter(fileTest))
    bwTest.write(s"${flattenHeader.mkString(",")}\n")
    testFeatures.foreach(row => {
      bwTest.write(s"${row.storage().array().mkString(",")}\n")
    })
    bwTest.close()
  }

  def main(args: Array[String]): Unit = {
    writePreprocessed(getClass.getClassLoader.getResource("house-prices-train.csv").getPath,
      getClass.getClassLoader.getResource("house-prices-test.csv").getPath,
      rowkeyName = "Id",
      labelName = "SalePrice")
  }
}
