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

import com.intel.analytics.bigdl.dataset.{LocalDataSet, MiniBatch}
import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, MklBlas}
import org.apache.log4j.Logger

object LocalValidator {
  val logger = Logger.getLogger(getClass)
}

/**
 * Validate a model on a single machine
 * Use given dataset with certain validation methods such as [[Top1Accuracy]]
 * as an argument of its `test` method
 *
 * @param model the model to be validated
 * @param dataSet the dataset used to validate a model
 */
class LocalValidator[T] private[optim](model: Module[T], dataSet: LocalDataSet[MiniBatch[T]])
  extends Validator[T, MiniBatch[T]](model, dataSet) {

  val logger = LocalValidator.logger
  private val coreNumber = Engine.coreNumber()

  private val subModelNumber = Engine.getEngineType match {
    case MklBlas => coreNumber
    case _ => throw new IllegalArgumentException
  }

  private val workingModels = (1 to subModelNumber).map(_ => model.cloneModule().evaluate()).toArray

  override def test(vMethods: Array[ValidationMethod[T]])
  : Array[(ValidationResult, ValidationMethod[T])] = {
    val dataIter = dataSet.data (train = false)
    var count = 0
    val vMethodsArr = (1 to subModelNumber).map(i => vMethods.map(_.clone())).toArray
    logger.info("model thread pool size is " + Engine.model.getPoolSize)
    dataIter.map(batch => {
      val stackSize = batch.size() / subModelNumber
      val extraSize = batch.size() % subModelNumber
      val parallelism = if (stackSize == 0) extraSize else subModelNumber
      val start = System.nanoTime()
      val result = Engine.default.invokeAndWait(
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
              validation(output.asInstanceOf[Tensor[T]], target)
            })
          }
        )
      ).reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      })
      count += batch.size()
      logger.info(s"[Validation] $count/${dataSet.size()} Throughput is ${
        batch.size() / ((System.nanoTime() - start) / 1e9)
      } record / sec")
      result
    }).reduce((left, right) => {
      left.zip(right).map { case (l, r) =>
        l + r
      }
    }).zip(vMethods)
  }
}
