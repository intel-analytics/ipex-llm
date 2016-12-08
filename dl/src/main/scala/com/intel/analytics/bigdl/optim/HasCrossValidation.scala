/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl.nn.abstractnn.{Activity, Module}
import com.intel.analytics.bigdl.optim.DistributedOptimizer.CachedModel
import com.intel.analytics.bigdl.tensor.Tensor
import org.apache.log4j.Logger
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ArrayBuffer

object HasCrossValidation{
  val logger = Logger.getLogger(getClass)
}

trait HasCrossValidation[@specialized(Float, Double) T] extends Serializable{
  import HasCrossValidation._
  private var testInterval: Int = 1

  def setTestInterval(testInterval: Int): this.type = {
    require(testInterval > 0)
    this.testInterval = testInterval
    this
  }

  val evaluateModels : RDD[CachedModel[T]]

  private var testDataSet: Option[DataSet[_, T]] = None

  def setTestDataSet(testDataSet: DataSet[_, T]): this.type = {
    this.testDataSet = Some(testDataSet)
    this
  }

  private val evalMethods = new ArrayBuffer[(String, (Tensor[T], Tensor[T]) => (Int, Int))]()

  def addEvaluation(name: String, evaluation: (Tensor[T], Tensor[T]) => (Int, Int))
  : this.type = {
    this.evalMethods.append((name, evaluation))
    this
  }

  def test(module: Module[_ <: Activity, _ <: Activity, T],
    iter: Int, wallClockNanoTime: Option[Long] = None): Array[Double] = {
    if (testDataSet.isDefined && iter % testInterval == 0) {
      evalMethods.map(evalM => {
        val evaluationBroadcast = testDataSet.get.getSparkContext().broadcast(evalM._2)
        val (correctSum, totalSum) = testDataSet.get.fetchAll().
          coalesce(evaluateModels.partitions.length, false).
          zipPartitions(evaluateModels)((data, cacheModelIter) => {
            val localModel = cacheModelIter.next().model
            localModel.evaluate()
            val localEvaluation = evaluationBroadcast.value
            Iterator.single(data.foldLeft((0, 0))((count, t) => {
              val result = localEvaluation(localModel.forward(t._1).toTensor[T], t._2)
              (count._1 + result._1, count._2 + result._2)
            }))
          }).reduce((a, b) => (a._1 + b._1, a._2 + b._2))

        val accuracy = correctSum.toDouble / totalSum
        if (wallClockNanoTime.isDefined) {
          logger.info(s"[Wall Clock ${wallClockNanoTime.get.toDouble / 1e9}s}] ${
            evalM._1
          } correct is $correctSum total is $totalSum")
          logger.info(s"[Wall Clock ${wallClockNanoTime.get.toDouble / 1e9}s}] ${
            evalM._1
          } accuracy is $accuracy")
        } else {
          logger.info(s"${evalM._1} correct is $correctSum total is $totalSum")
          logger.info(s"${evalM._1} cross validation result is $accuracy")
        }
        accuracy
      }).toArray
    } else {
      Array[Double]()
    }
  }
}
