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
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.dataset.{DistributedDataSet, LocalDataSet, MiniBatch}
import org.apache.log4j.Logger

/**
 * [[Validator]] is an abstract class which is used to test a model automatically
 * with some certain validation methods such as [[Top1Accuracy]], as an argument of
 * its `test` method.
 *
 * @param model the model to be validated
 * @param dataSet the data set used to validate a model
 * @tparam T numeric type, which can be [[Float]] or [[Double]]
 * @tparam D the type of elements in DataSet, such as [[MiniBatch]]
 */
abstract class Validator[T, D](
  model: Module[T],
  dataSet: DataSet[D]
) {
  def test(vMethods: Array[ValidationMethod[T]]): Array[(ValidationResult, ValidationMethod[T])]
}
@deprecated(
  "Validator(model, dataset) is deprecated. Please use model.evaluate instead",
  "0.2.0")
object Validator {
  private val logger = Logger.getLogger(getClass)

  def apply[T, D](model: Module[T], dataset: DataSet[D]): Validator[T, D] = {
    logger.warn("Validator(model, dataset) is deprecated. Please use model.evaluate instead")
    dataset match {
      case d: DistributedDataSet[_] =>
        new DistriValidator[T](
          model = model,
          dataSet = d.asInstanceOf[DistributedDataSet[MiniBatch[T]]]
        ).asInstanceOf[Validator[T, D]]
      case d: LocalDataSet[_] =>
        new LocalValidator[T](
          model = model,
          dataSet = d.asInstanceOf[LocalDataSet[MiniBatch[T]]]
        ).asInstanceOf[Validator[T, D]]
      case _ =>
        throw new UnsupportedOperationException
    }
  }
}
