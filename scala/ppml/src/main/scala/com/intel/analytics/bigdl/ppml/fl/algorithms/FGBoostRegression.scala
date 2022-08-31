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

package com.intel.analytics.bigdl.ppml.fl.algorithms

import com.intel.analytics.bigdl.dllib.optim.MAE
import com.intel.analytics.bigdl.ppml.fl.fgboost.FGBoostModel
import com.intel.analytics.bigdl.ppml.fl.fgboost.common.RegressionTree
import org.apache.log4j.LogManager

import java.io.{BufferedWriter, File, FileWriter}
import scala.io.Source
import scala.util.parsing.json.{JSON, JSONObject}

/**
 * FGBoost regression algorithm
 * @param learningRate learning rate
 * @param maxDepth max depth of boosting tree
 * @param minChildSize
 */
class FGBoostRegression(learningRate: Float = 0.005f,
                        maxDepth: Int = 6,
                        minChildSize: Int = 1,
                        serverModelPath: String = null)
  extends FGBoostModel(continuous = true,
    learningRate = learningRate,
    maxDepth = maxDepth,
    minChildSize = minChildSize,
    validationMethods = Array(new MAE()),
    serverModelPath = serverModelPath) {

  def toJSON(): JSONObject = {
    JSONObject(Map(
      "maxDepth" -> maxDepth,
      "learningRate" -> learningRate,
      "minChildSize" -> minChildSize,
      "trees" -> JSONObject(scala.collection.SortedMap(
        trees.zipWithIndex.sortBy(_._2).map(v => (v._2.toString, v._1.toJson())): _*)
      (Ordering.by(_.toInt)).toMap)))
  }
  def saveModel(dest: String): Unit = {
    val file = new File(dest)
    val bufferedWriter = new BufferedWriter(new FileWriter(file))
    bufferedWriter.write(this.toJSON().toString())
    bufferedWriter.close()
    logger.info(s"saved model to $dest")
  }
}

object FGBoostRegression {
  val logger = LogManager.getLogger(getClass)
  def fromJson(str: String): FGBoostRegression = {
    val json = JSON.parseRaw(str).get.asInstanceOf[JSONObject]
    val learningRate = json.obj.get("learningRate").get.asInstanceOf[Double].toFloat
    val maxDepth = json.obj.get("maxDepth").get.asInstanceOf[Double].toInt
    val minChildSize = json.obj.get("minChildSize").get.asInstanceOf[Double].toInt
    val gbr = new FGBoostRegression(learningRate, maxDepth, minChildSize)
    val trees = json.obj.get("trees").get.asInstanceOf[JSONObject].obj.mapValues{v =>
      RegressionTree.fromJson(v.asInstanceOf[JSONObject])
    }
    trees.toArray.sortBy(_._1.toInt).foreach{t =>
      gbr.trees.enqueue(t._2)
    }
    logger.info(s"FGBoost Regression model loaded, tree number: ${gbr.trees.size}")
    gbr
  }
  def loadModel(src: String): FGBoostRegression = {
    val jsonStr = Source.fromFile(src, "utf-8").mkString
    logger.info(s"loading model from $src")
    fromJson(jsonStr)
  }
}
