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

class DistriValidator[T] private[optim](
  model: Module[T],
  dataSet: DistributedDataSet[MiniBatch[T]]
) extends Validator[T, MiniBatch[T]](model, dataSet) {

  override def test(vMethods: Array[ValidationMethod[T]])
  : Array[(ValidationResult, ValidationMethod[T])] = {

    val rdd = dataSet.data(train = false)
    val broadcastModel = rdd.sparkContext.broadcast(model.evaluate())
    val _subModelNumber = Engine.getEngineType match {
      case MklBlas => Engine.coreNumber()
      case _ => throw new IllegalArgumentException
    }
    val nExecutor = Engine.nodeNumber()
    val executorCores = Engine.coreNumber()
    rdd.mapPartitions(dataIter => {
      Engine.setNodeAndCore(nExecutor, executorCores)
      val localModel = broadcastModel.value
      logger.info("model thread pool size is " + Engine.model.getPoolSize)
      val workingModels = (1 to _subModelNumber)
        .map(_ => localModel.cloneModule().evaluate()).toArray
      dataIter.map(batch => {
        require(batch.data.size(1) == batch.labels.size(1))
        val stackSize = batch.data.size(1) / _subModelNumber
        val extraSize = batch.data.size(1) % _subModelNumber
        val parallelism = if (stackSize == 0) extraSize else _subModelNumber
        Engine.default.invokeAndWait(
          (0 until parallelism).map(b =>
            () => {
              val offset = b * stackSize + math.min(b, extraSize)
              val length = stackSize + (if (b < extraSize) 1 else 0)
              val input = batch.data.narrow(1, offset + 1, length)
              val target = batch.labels.narrow(1, offset + 1, length)
              val output = workingModels(b).forward(input)
              vMethods.map(validation => {
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
