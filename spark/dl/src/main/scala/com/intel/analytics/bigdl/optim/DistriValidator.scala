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

package com.intel.analytics.bigdl.optim

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, MiniBatch}
import com.intel.analytics.bigdl.optim.DistriValidator._
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}
import org.apache.log4j.Logger

object DistriValidator {
  val logger = Logger.getLogger(this.getClass)
}

/**
 * Validate model on a distributed cluster.
 *
 * @param model model to be validated
 * @param dataSet validation dataset
 */
class DistriValidator[T] private[optim](
  model: Module[T],
  dataSet: DistributedDataSet[MiniBatch[T]]
) extends Validator[T, MiniBatch[T]](model, dataSet) {

  /**
   * Applies vMethods to the model and validation dataset.
   * @param vMethods
   * @return
   */
  override def test(vMethods: Array[ValidationMethod[T]])
  : Array[(ValidationResult, ValidationMethod[T])] = {

    val rdd = dataSet.data(train = false)
    val broadcastModel = rdd.sparkContext.broadcast(model.evaluate(), vMethods)
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()
    rdd.mapPartitions(dataIter => {
      Engine.setNodeAndCore(nExecutor, executorCores)
      val localModel = broadcastModel.value._1
      val localMethod = broadcastModel.value._2
      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val workingModels = (1 to _subModelNumber)
        .map(_ => localModel.cloneModule().evaluate()).toArray
      val vMethodsArr = (1 to _subModelNumber).map(i => localMethod.map(_.clone())).toArray
      dataIter.map(batch => {
        val stackSize = batch.size() / _subModelNumber
        val extraSize = batch.size() % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize) + 1
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val currentMiniBatch = batch.slice(offset, length)
              val input = currentMiniBatch.getInput()
              val target = currentMiniBatch.getTarget()
              val output = workingModels(b).forward(input)
              val validatMethods = vMethodsArr(b)
              validatMethods.map(validation => {
                validation(output, target)
              })
            }
          )
        ).reduce((left, right) => {
          left.zip(right).map { case (l, r) =>
            l + r
          }
        })
      })
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods)
  }
}
